"""Evaluation utilities for behavior cloning policies."""
from __future__ import annotations

import copy
import io
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import gym_pusht  # noqa: F401
import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch
from imitation.data import Normalizer
from imitation.data import ROBOMIMIC_OBS_KEYS
from imitation.model import BasePolicy
from PIL import Image

import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.obs_utils as ObsUtils
import wandb


PUSHT_ENV_ID = "gym_pusht/PushT-v0"
NUM_EVAL_EPISODES = 100
ROBOMIMIC_HORIZON = 700


class Logger:
    """Logger for logging metrics."""

    rows: list[dict]

    CSV_DISALLOWED_TYPES = (wandb.Image, wandb.Video, wandb.Histogram)

    def __init__(self, path: Path):
        if path.exists():
            raise FileExistsError(f"Log directory {path} already exists.")
        path.mkdir(parents=True)
        self.path = path
        self.csv_path = path / "log.csv"
        self.rows = []

    def log(self, row: dict[str, Any], step: int) -> None:
        row["step"] = step
        wandb.log(row, step=step)
        self.rows.append(copy.deepcopy(row))

    def dump_logs(self) -> None:
        fields = set(
            k
            for row in self.rows
            for k, v in row.items()
            if not isinstance(v, self.CSV_DISALLOWED_TYPES)
        )
        fields = list(fields)
        with self.csv_path.open("w") as f:
            f.write(",".join(fields) + "\n")
            for row in self.rows:
                filtered_row = [str(row.get(field, "")) for field in fields]
                f.write(",".join(filtered_row) + "\n")
        wandb_dir = Path(wandb.run.dir).parent
        wandb.finish()
        shutil.copytree(wandb_dir, self.path / "wandb_out")


def resize_frame(frame: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    image = Image.fromarray(frame)
    resized = image.resize(size, resample=Image.BILINEAR)
    return np.asarray(resized)


def encode_video(frames: list[np.ndarray], fps: int = 20) -> wandb.Video | None:
    if not frames:
        return None

    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        with imageio.get_writer(
            tmp_path,
            fps=fps,
            codec="libx264",
            macro_block_size=1,
        ) as writer:
            for frame in frames:
                writer.append_data(frame)
        with open(tmp_path, "rb") as f:
            video_bytes = f.read()
        return wandb.Video(io.BytesIO(video_bytes), format="mp4")
    finally:
        try:
            os.remove(tmp_path)
        except FileNotFoundError:
            pass


def log_checkpoint_artifact(model: BasePolicy, step: int) -> None:
    if wandb.run is None:
        raise RuntimeError("wandb.init did not create a run.")

    run_dir = Path(wandb.run.dir)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"checkpoint_step_{step}.pkl"
    torch.save(model, checkpoint_path)

    artifact = wandb.Artifact(
        name=f"policy-checkpoint-{wandb.run.id}",
        type="model",
        metadata={"step": step},
    )
    artifact.add_file(checkpoint_path.as_posix(), name=checkpoint_path.name)
    wandb.log_artifact(artifact)


def get_action_chunk(
    model: BasePolicy,
    obs: np.ndarray,
    normalizer: Normalizer,
    device: torch.device,
    flow_num_steps: int,
) -> np.ndarray:
    state = torch.from_numpy(normalizer.normalize_state(obs)).float().to(device)
    with torch.no_grad():
        pred_chunk = (
            model.sample_actions(state.unsqueeze(0), num_steps=flow_num_steps)
            .cpu()
            .numpy()[0]
        )
    action_chunk = normalizer.denormalize_action(pred_chunk)
    return action_chunk


def run_eval_pusht(
    model: BasePolicy,
    normalizer: Normalizer,
    device: torch.device,
    chunk_size: int,
    video_size: tuple[int, int],
    num_video_episodes: int,
    flow_num_steps: int,
) -> tuple[list[bool], list[float], list[wandb.Video]]:
    successes, rewards, videos = [], [], []
    env = gym.make(PUSHT_ENV_ID, obs_type="state", render_mode="rgb_array")
    action_low = env.action_space.low
    action_high = env.action_space.high
    for ep_idx in range(NUM_EVAL_EPISODES):
        obs, _ = env.reset(seed=ep_idx)
        done = False
        chunk_index = chunk_size
        action_chunk: np.ndarray | None = None
        frames: list[np.ndarray] = []
        max_reward = 0.0
        save_video = ep_idx < num_video_episodes
        terminated = False
        while not done:
            if action_chunk is None or chunk_index >= chunk_size:
                action_chunk = get_action_chunk(
                    model, obs, normalizer, device, flow_num_steps
                )
                action_chunk = np.clip(action_chunk, action_low, action_high)
                chunk_index = 0
            action = action_chunk[chunk_index]
            obs, reward, terminated, truncated, _ = env.step(action.astype(np.float32))
            if save_video:
                frame = env.render()
                frame = resize_frame(frame, video_size)
                frames.append(frame)
            max_reward = max(max_reward, float(reward))
            done = terminated or truncated
            chunk_index += 1
        successes.append(terminated)
        rewards.append(max_reward)
        if save_video:
            video = encode_video(frames, fps=20)
            if video is not None:
                videos.append(video)
    env.close()
    return successes, rewards, videos


def run_eval_robomimic(
    model: BasePolicy,
    dataset_path: Path,
    normalizer: Normalizer,
    device: torch.device,
    chunk_size: int,
    video_size: tuple[int, int],
    num_video_episodes: int,
    flow_num_steps: int,
) -> tuple[list[bool], list[float], list[wandb.Video]]:
    obs_spec = dict(
        obs=dict(
            low_dim=["robot0_eef_pos"],
            rgb=["agentview_image"],
        ),
    )
    ObsUtils.initialize_obs_utils_with_obs_specs(obs_modality_specs=obs_spec)
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        env_name=env_meta["env_name"],
        render=False,
        render_offscreen=True,
        use_image_obs=False,
    )  # type: ignore
    successes, rewards, videos = [], [], []
    for ep_idx in range(NUM_EVAL_EPISODES):
        obs = env.reset()  # TODO: Use seed ep_idx to reset
        state_dict = env.get_state()
        # hack that is necessary for robosuite tasks for deterministic action playback
        obs = env.reset_to(state_dict)
        obs = np.concat(tuple([obs[key] for key in ROBOMIMIC_OBS_KEYS]))
        chunk_index = chunk_size
        action_chunk: np.ndarray | None = None
        frames: list[np.ndarray] = []
        max_reward = 0.0
        save_video = ep_idx < num_video_episodes
        step_num = 0
        while (
            not env.is_done()
            and not env.is_success()["task"]
            and step_num < ROBOMIMIC_HORIZON
        ):
            if action_chunk is None or chunk_index >= chunk_size:
                action_chunk = get_action_chunk(
                    model, obs, normalizer, device, flow_num_steps
                )
                action_chunk = np.clip(action_chunk, -1, 1)
                chunk_index = 0
            action = action_chunk[chunk_index]
            obs, reward, _, _ = env.step(action.astype(np.float32))
            obs = np.concat(tuple([obs[key] for key in ROBOMIMIC_OBS_KEYS]))
            if save_video:
                frame = env.render(
                    "rgb_array", height=video_size[1], width=video_size[0]
                )
                frames.append(frame)
            max_reward = max(max_reward, float(reward))
            chunk_index += 1
            step_num += 1
        successes.append(env.is_success()["task"])
        rewards.append(max_reward)
        if save_video:
            video = encode_video(frames, fps=20)
            if video is not None:
                videos.append(video)
    return successes, rewards, videos


def evaluate_policy(
    dataset_path: Path,
    model: BasePolicy,
    normalizer: Normalizer,
    device: torch.device,
    chunk_size: int,
    video_size: tuple[int, int],
    num_video_episodes: int,
    flow_num_steps: int,
    step: int,
    logger: Logger,
) -> None:
    """Evaluate a policy in environment and log results to Weights & Biases.

    This function runs a fixed number of evaluation episodes in a gym
    environment using the provided policy. It normalizes observations with the
    given normalizer, requests a chunk of actions from the policy (optionally
    using multiple sampling steps for flow-based policies), and executes those
    actions in the environment until each episode terminates.

    Metrics:
        - Logs the mean of per-episode maximum reward.
        - Optionally logs rendered rollout videos for the first
          ``num_video_episodes`` episodes.

    Checkpointing:
        - Saves the policy as a ``.pkl`` file and uploads it as a W&B artifact
          tagged with the current training step.

    Args:
        model: The policy to evaluate.
        normalizer: Normalizer used to scale states and actions.
        device: Device on which to run policy inference.
        chunk_size: Number of actions to generate per policy call.
        video_size: (width, height) for rendered rollout videos.
        num_video_episodes: How many episodes to record videos for.
        flow_num_steps: Number of denoising steps used by flow policies.
        step: Training step used for logging and artifact metadata.
        logger: Logger for logging metrics.
    """
    model.eval()
    if dataset_path.suffix == ".zarr":  # PushT
        successes, rewards, videos = run_eval_pusht(
            model,
            normalizer,
            device,
            chunk_size,
            video_size,
            num_video_episodes,
            flow_num_steps,
        )
    else:  # robomimic
        successes, rewards, videos = run_eval_robomimic(
            model,
            dataset_path,
            normalizer,
            device,
            chunk_size,
            video_size,
            num_video_episodes,
            flow_num_steps,
        )
    log_data: dict[str, float | wandb.Video] = {
        "eval/mean_reward": float(np.mean(rewards)),
        "eval/success_rate": sum(successes) / len(successes),
    }
    for idx, video in enumerate(videos):
        log_data[f"eval/rollout_ep{idx}"] = video
    logger.log(log_data, step=step)
    log_checkpoint_artifact(model, step=step)
