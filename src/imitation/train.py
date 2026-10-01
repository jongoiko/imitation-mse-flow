"""Train and evaluate a behavior cloning policy."""
from __future__ import annotations

import copy
from dataclasses import asdict
from dataclasses import dataclass
from dataclasses import field
from datetime import datetime
from pathlib import Path
from typing import Any
from typing import Literal

import numpy as np
import torch
import tyro
from imitation.data import ActionChunkDataset
from imitation.data import download_dataset
from imitation.data import load_demonstrations
from imitation.data import Normalizer
from imitation.ema import EMAModel
from imitation.evaluation import evaluate_policy
from imitation.evaluation import log_checkpoint_artifact
from imitation.evaluation import Logger
from imitation.model import BasePolicyModel
from imitation.model import build_policy
from imitation.model import FlowPolicy
from imitation.model import MSEPolicy
from torch.utils.data import DataLoader

import wandb

LOGDIR_PREFIX = "exp"


@dataclass
class TrainConfig:
    # The task that the policy is trained and evaluated on.
    task: Literal[
        "pusht",
        "robomimic/lift",
        "robomimic/can",
        "robomimic/square",
        "robomimic/tool_hang",
        "robomimic/transport",
    ] = "pusht"
    # The path to download the dataset to.
    data_dir: Path = Path("data")
    # Path of checkpoint to resume training from. If not provided, a policy is trained from scratch.
    checkpoint: Path | None = None
    # The policy type -- either MSE or flow.
    policy: MSEPolicy | FlowPolicy = field(default_factory=MSEPolicy)
    # The predicted action chunk size. Must be >= exec_chunk_size.
    pred_chunk_size: int = 8
    # The executed action chunk size. Must be <= pred_chunk_size.
    exec_chunk_size: int = 8
    # The horizon of past observations/states to pass as input to the policy.
    obs_horizon: int = 1
    # Whether to convert the 3D rotation component of the action space to the 6D
    # representation of Zhou et al.
    rot_to_6d: bool = False
    # Whether to use an Exponential Moving Average (EMA) of model weights for
    # policy evaluation.
    use_ema: bool = False
    # Whether to use Automatic Mixed Precision (AMP) during training.
    use_amp: bool = True
    # The batch size.
    batch_size: int = 512
    # The AdamW learning rate.
    lr: float = 3e-4
    # The AdamW weight decay.
    weight_decay: float = 0.0
    # The number and size of hidden layers: if using an MLP, then these are the
    # hidden layer sizes, and if using a 1D UNet, these are the downsampling
    # dimensions.
    hidden_dims: tuple[int, ...] = (256, 256, 256)
    # The number of epochs to train for.
    num_epochs: int = 3000
    # How many episodes to run at each policy evaluation.
    num_eval_episodes: int = 200
    # How often to run evaluation, measured in training steps.
    eval_interval: int = 100_000
    # How many videos to record during evaluation.
    num_video_episodes: int = 5
    # The size of recorded rollout videos.
    video_size: tuple[int, int] = (256, 256)
    # How often to log training metrics, measured in training steps.
    log_interval: int = 200
    # Random seed.
    seed: int = 42
    # WandB project name.
    wandb_project: str = "imitation-mse-flow"
    # Experiment name suffix for logging and WandB.
    exp_name: str | None = None


def parse_train_config(
    args: list[str] | None = None,
    *,
    defaults: TrainConfig | None = None,
    description: str = "Train a behavior cloning policy.",
) -> TrainConfig:
    defaults = defaults or TrainConfig()
    return tyro.cli(
        TrainConfig,
        config=(
            tyro.conf.UsePythonSyntaxForLiteralCollections,
            tyro.conf.FlagConversionOff,
        ),
        args=args,
        default=defaults,
        description=description,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def config_to_dict(config: TrainConfig) -> dict[str, Any]:
    data = asdict(config)
    for key, value in data.items():
        if isinstance(value, Path):
            data[key] = str(value)
    return data


def make_checkpoint(
    epoch: int,
    model: BasePolicyModel,
    ema_model: EMAModel | None,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
) -> dict:
    return {
        "epoch": epoch,
        "model": model,
        "ema_model": ema_model,
        "optimizer": optimizer,
        "lr_scheduler": lr_scheduler,
    }


def restore_checkpoint(
    checkpoint: Any,
) -> tuple[
    int,
    BasePolicyModel,
    EMAModel | None,
    torch.optim.Optimizer,
    torch.optim.lr_scheduler.LRScheduler,
]:
    return tuple(
        checkpoint[key]
        for key in ["epoch", "model", "ema_model", "optimizer", "lr_scheduler"]
    )


def run_training_loop(
    config: TrainConfig,
    epoch_idx: int,
    dataset_path: Path,
    loader: DataLoader,
    model: BasePolicyModel,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    normalizer: Normalizer,
    ema_model: EMAModel | None,
    logger: Logger,
    device: torch.device,
) -> int:
    total_training_steps = 0
    model.train()
    compute_loss = torch.compile(model.compute_loss)
    grad_scaler = torch.amp.GradScaler(enabled=config.use_amp)
    for epoch_idx in range(epoch_idx, config.num_epochs):
        for batch in loader:
            model.train()
            state, action_chunk = batch
            optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=config.use_amp
            ):
                loss = compute_loss(state.to(device), action_chunk.to(device))
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            if ema_model is not None:
                ema_model.step(model)
            total_training_steps += 1
            if total_training_steps % config.eval_interval == 0:
                eval_model: BasePolicyModel = (
                    ema_model.averaged_model if ema_model is not None else model
                )  # type: ignore
                eval_model.eval()
                num_flow_steps = (
                    config.policy.flow_num_steps
                    if isinstance(config.policy, FlowPolicy)
                    else 0
                )
                evaluate_policy(
                    dataset_path,
                    eval_model,
                    config.num_eval_episodes,
                    normalizer,
                    device,
                    config.exec_chunk_size,
                    config.video_size,
                    config.num_video_episodes,
                    num_flow_steps,
                    total_training_steps,
                    epoch_idx,
                    logger,
                    config.rot_to_6d,
                )
                ckpt = make_checkpoint(
                    epoch_idx,
                    model,
                    ema_model,
                    optimizer,
                    lr_scheduler,
                )  # type: ignore
                log_checkpoint_artifact(ckpt, total_training_steps)
            if total_training_steps % config.log_interval == 0:
                logger.log(
                    {"train/loss": float(loss.item()), "epoch": epoch_idx},
                    step=total_training_steps,
                )
        lr_scheduler.step()
    return total_training_steps


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = download_dataset(config.task, config.data_dir)
    states, actions, episode_ends = load_demonstrations(
        dataset_path, config.pred_chunk_size, axis_angle_to_rot6d=config.rot_to_6d
    )
    normalizer = Normalizer.from_data(states, actions)

    dataset = ActionChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.pred_chunk_size,
        observation_horizon=config.obs_horizon,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    if config.checkpoint is None:
        model = build_policy(
            config.policy,
            state_dim=states.shape[1],
            action_dim=actions.shape[1],
            chunk_size=config.pred_chunk_size,
            observation_horizon=config.obs_horizon,
            hidden_dims=config.hidden_dims,
        ).to(device)
        ema_model = EMAModel(copy.deepcopy(model)) if config.use_ema else None
        optimizer = torch.optim.AdamW(
            model.parameters(), config.lr, weight_decay=config.weight_decay
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, config.num_epochs
        )
        epoch_idx = 0
    else:
        ckpt = torch.load(config.checkpoint, weights_only=False)
        epoch_idx, model, ema_model, optimizer, lr_scheduler = restore_checkpoint(ckpt)
        model = model.to(device)

    model.train()
    model: BasePolicyModel = torch.compile(model)  # type: ignore
    total_trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print(f"Policy has {total_trainable_params:,} trainable parameters")

    exp_name = f"seed_{config.seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    if config.exp_name is not None:
        exp_name += f"_{config.exp_name}"
    log_dir = Path(LOGDIR_PREFIX) / exp_name
    wandb.init(
        project=config.wandb_project, config=config_to_dict(config), name=exp_name
    )
    logger = Logger(log_dir)

    total_training_steps = run_training_loop(
        config,
        epoch_idx,
        dataset_path,
        loader,
        model,
        optimizer,
        lr_scheduler,
        normalizer,
        ema_model,
        logger,
        device,
    )
    eval_model: BasePolicyModel = (
        ema_model.averaged_model if ema_model is not None else model
    )  # type: ignore
    eval_model.eval()
    evaluate_policy(
        dataset_path,
        eval_model,
        config.num_eval_episodes,
        normalizer,
        device,
        config.exec_chunk_size,
        config.video_size,
        config.num_video_episodes,
        config.policy.flow_num_steps if isinstance(config.policy, FlowPolicy) else 0,
        total_training_steps,
        config.num_epochs,
        logger,
        config.rot_to_6d,
    )
    ckpt = make_checkpoint(
        config.num_epochs,
        model,
        ema_model,
        optimizer,
        lr_scheduler,
    )
    log_checkpoint_artifact(ckpt, total_training_steps)
    logger.dump_logs()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
