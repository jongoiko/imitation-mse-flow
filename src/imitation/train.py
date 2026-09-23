"""Train and evaluate a behavior cloning policy."""
from __future__ import annotations

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
from imitation.evaluation import evaluate_policy
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
    # The policy type -- either MSE or flow.
    policy: MSEPolicy | FlowPolicy = field(default_factory=MSEPolicy)
    # The action chunk size.
    chunk_size: int = 8
    # The horizon of past observations/states to pass as input to the policy.
    obs_horizon: int = 1
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


def run_training_loop(
    config: TrainConfig,
    dataset_path: Path,
    loader: DataLoader,
    model: BasePolicyModel,
    normalizer: Normalizer,
    logger: Logger,
    device: torch.device,
) -> int:
    optimizer = torch.optim.AdamW(
        model.parameters(), config.lr, weight_decay=config.weight_decay
    )
    total_training_steps = 0
    model.train()
    compute_loss = torch.compile(model.compute_loss)
    for _ in range(config.num_epochs):
        for batch in loader:
            state, action_chunk = batch
            optimizer.zero_grad()
            loss = compute_loss(state.to(device), action_chunk.to(device))
            loss.backward()
            optimizer.step()
            total_training_steps += 1
            if total_training_steps % config.eval_interval == 0:
                model.eval()
                num_flow_steps = (
                    config.policy.flow_num_steps
                    if isinstance(config.policy, FlowPolicy)
                    else 0
                )
                evaluate_policy(
                    dataset_path,
                    model,
                    normalizer,
                    device,
                    config.chunk_size,
                    config.video_size,
                    config.num_video_episodes,
                    num_flow_steps,
                    total_training_steps,
                    logger,
                )
                model.train()
            if total_training_steps % config.log_interval == 0:
                logger.log(
                    {"train/loss": float(loss.item())}, step=total_training_steps
                )
    return total_training_steps


def run_training(config: TrainConfig) -> None:
    set_seed(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    dataset_path = download_dataset(config.task, config.data_dir)
    states, actions, episode_ends = load_demonstrations(dataset_path)
    normalizer = Normalizer.from_data(states, actions)

    dataset = ActionChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        observation_horizon=config.obs_horizon,
        normalizer=normalizer,
    )

    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
    )

    model = build_policy(
        config.policy,
        state_dim=states.shape[1],
        action_dim=actions.shape[1],
        chunk_size=config.chunk_size,
        observation_horizon=config.obs_horizon,
        hidden_dims=config.hidden_dims,
    ).to(device)
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
        config, dataset_path, loader, model, normalizer, logger, device
    )
    model.eval()
    evaluate_policy(
        dataset_path,
        model,
        normalizer,
        device,
        config.chunk_size,
        config.video_size,
        config.num_video_episodes,
        config.policy.flow_num_steps if isinstance(config.policy, FlowPolicy) else 0,
        total_training_steps,
        logger,
    )
    logger.dump_logs()


def main() -> None:
    config = parse_train_config()
    run_training(config)


if __name__ == "__main__":
    main()
