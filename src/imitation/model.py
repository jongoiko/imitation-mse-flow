import abc
from typing import Literal
from typing import TypeAlias

import torch
from einops import pack
from einops import rearrange
from torch import nn


class BasePolicy(nn.Module, metaclass=abc.ABCMeta):
    """Base class for action chunking policies."""

    def __init__(self, state_dim: int, action_dim: int, chunk_size: int) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size

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


class MSEPolicy(BasePolicy):
    """Predicts action chunks with an MSE loss."""

    mlp: nn.Sequential
    chunk_size: int

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.mlp = make_relu_mlp(state_dim, chunk_size * action_dim, hidden_dims)
        self.chunk_size = chunk_size

    def compute_loss(
        self,
        state: torch.Tensor,
        action_chunk: torch.Tensor,
    ) -> torch.Tensor:
        pred_action_chunk = rearrange(
            self.mlp(state), "b (t a) -> b t a", t=self.chunk_size
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
        pred_action_chunk = rearrange(
            self.mlp(state), "b (t a) -> b t a", t=self.chunk_size
        )
        return pred_action_chunk


class FlowMatchingPolicy(BasePolicy):
    """Predicts action chunks with a flow matching loss."""

    mlp: nn.Sequential
    chunk_size: int
    action_dim: int

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        chunk_size: int,
        hidden_dims: tuple[int, ...] = (128, 128),
    ) -> None:
        super().__init__(state_dim, action_dim, chunk_size)
        self.mlp = make_relu_mlp(
            state_dim + chunk_size * action_dim + 1,
            chunk_size * action_dim,
            hidden_dims,
        )
        self.chunk_size = chunk_size
        self.action_dim = action_dim

    def _predict_velocity(
        self, state: torch.Tensor, time: torch.Tensor, action_chunk: torch.Tensor
    ) -> torch.Tensor:
        policy_input, _ = pack([state, time, action_chunk], "b *")
        return rearrange(self.mlp(policy_input), "b (t a) -> b t a", t=self.chunk_size)

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
        pred_velocity = self._predict_velocity(state, time, interpolation)
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
        action_chunk = torch.randn(batch_size, self.chunk_size, self.action_dim).to(
            device
        )
        with torch.no_grad():
            for time in torch.linspace(0, 1, num_steps + 1)[:-1].to(device):
                pred_velocity = self._predict_velocity(
                    state, time.repeat(batch_size), action_chunk
                )
                action_chunk += pred_velocity / num_steps
        return action_chunk


PolicyType: TypeAlias = Literal["mse", "flow"]


def build_policy(
    policy_type: PolicyType,
    *,
    state_dim: int,
    action_dim: int,
    chunk_size: int,
    hidden_dims: tuple[int, ...] = (128, 128),
) -> BasePolicy:
    if policy_type == "mse":
        return MSEPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    if policy_type == "flow":
        return FlowMatchingPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            chunk_size=chunk_size,
            hidden_dims=hidden_dims,
        )
    raise ValueError(f"Unknown policy type: {policy_type}")
