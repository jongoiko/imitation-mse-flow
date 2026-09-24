import abc
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from imitation.flow_architectures import BaseVelocityPredictor
from imitation.flow_architectures import ConditionalUnet1D
from imitation.flow_architectures import MLPVelocityPredictor
from torch import nn


class BasePolicyModel(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    state_dim: int
    action_dim: int
    action_chunk_horizon: int
    observation_horizon: int

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_chunk_horizon: int,
        observation_horizon: int,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_chunk_horizon = action_chunk_horizon
        self.observation_horizon = observation_horizon

    @abc.abstractmethod
    def compute_loss(
        self, state: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        """Compute training loss for a batch."""

    @abc.abstractmethod
    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,  # only applicable for flow policy
    ) -> torch.Tensor:
        """Generate a chunk of actions with shape (batch, chunk_size, action_dim)."""


def make_relu_mlp(
    input_dim: int, output_dim: int, hidden_dims: tuple[int, ...]
) -> nn.Sequential:
    layer_dims = [input_dim] + list(hidden_dims) + [output_dim]
    layers = []
    for in_dim, out_dim in zip(layer_dims[:-1], layer_dims[1:]):
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
    return nn.Sequential(*layers[:-1])


class MSEPolicyModel(BasePolicyModel):
    """Predicts action chunks with an MSE loss."""

    mlp: nn.Sequential

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        action_chunk_horizon: int,
        observation_horizon: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(
            state_dim, action_dim, action_chunk_horizon, observation_horizon
        )
        self.mlp = make_relu_mlp(
            state_dim, action_chunk_horizon * action_dim, hidden_dims
        )

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_action_chunk = rearrange(
            self.mlp(state), "b (t a) -> b t a", t=self.action_chunk_horizon
        )
        batch_size = state.shape[0]
        loss = nn.functional.mse_loss(pred_action_chunk, action_chunk, reduction="sum")
        return loss / batch_size

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        with torch.no_grad():
            pred_action_chunk = rearrange(
                self.mlp(state), "b (t a) -> b t a", t=self.action_chunk_horizon
            )
        return pred_action_chunk


class FlowMatchingPolicyModel(BasePolicyModel):
    """Predicts action chunks with a flow matching loss."""

    vel_predictor: BaseVelocityPredictor

    def __init__(
        self,
        vel_predictor: BaseVelocityPredictor,
        state_dim: int,
        action_dim: int,
        action_chunk_horizon: int,
        observation_horizon: int,
    ) -> None:
        super().__init__(
            state_dim, action_dim, action_chunk_horizon, observation_horizon
        )
        self.vel_predictor = vel_predictor

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        device = state.device
        noise = torch.randn(*action_chunk.shape).to(device)
        batch_size = state.shape[0]
        time = torch.rand(batch_size).to(device)
        interpolation = (
            time[:, None, None] * action_chunk + (1 - time[:, None, None]) * noise
        )
        pred_velocity = self.vel_predictor.predict_velocity(state, time, interpolation)
        loss = nn.functional.mse_loss(
            pred_velocity, action_chunk - noise, reduction="sum"
        )
        return loss / batch_size

    def sample_actions(
        self,
        state: torch.Tensor,
        *,
        num_steps: int = 10,
    ) -> torch.Tensor:
        batch_size = state.shape[0]
        device = state.device
        action_chunk = torch.randn(
            batch_size, self.action_chunk_horizon, self.action_dim
        ).to(device)
        with torch.no_grad():
            for time in torch.linspace(0, 1, num_steps + 1)[:-1].to(device):
                pred_velocity = self.vel_predictor.predict_velocity(
                    state, time.repeat(batch_size), action_chunk
                )
                action_chunk += pred_velocity / num_steps
        return action_chunk


@dataclass
class FlowPolicy:
    """Flow policy."""

    # The number of denoising steps to use for the flow policy.
    flow_num_steps: int = 10
    # Architecture.
    architecture: Literal["mlp", "unet"] = "mlp"
    # The dimension of the time embedding vectors (for UNet).
    time_embed_dim: int = 128
    # The size of the 1D convolution kernels (for UNet).
    conv_kernel_size: int = 3
    # The number of groups for GroupNorm layers (for UNet).
    groupnorm_n_groups: int = 8
    # Whether to predict or not the scale (\gamma) for FiLM conditioning (for UNet).
    cond_predict_scale: bool = True


@dataclass
class MSEPolicy:
    """MSE (mean squared error) policy."""


def build_policy(
    policy_config: MSEPolicy | FlowPolicy,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    observation_horizon: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicyModel:
    state_dim = state_dim * observation_horizon
    if isinstance(policy_config, MSEPolicy):
        return MSEPolicyModel(
            state_dim=state_dim,
            action_dim=action_dim,
            action_chunk_horizon=chunk_size,
            observation_horizon=observation_horizon,
            hidden_dims=hidden_dims,
        )
    if policy_config.architecture == "mlp":
        mlp = make_relu_mlp(
            state_dim + chunk_size * action_dim + 1,
            chunk_size * action_dim,
            hidden_dims,
        )
        predictor = MLPVelocityPredictor(mlp)
    else:
        predictor = ConditionalUnet1D(
            action_dim,
            state_dim,
            policy_config.time_embed_dim,
            hidden_dims,
            policy_config.conv_kernel_size,
            policy_config.groupnorm_n_groups,
            policy_config.cond_predict_scale,
        )
    return FlowMatchingPolicyModel(
        predictor,
        state_dim=state_dim,
        action_dim=action_dim,
        action_chunk_horizon=chunk_size,
        observation_horizon=observation_horizon,
    )
