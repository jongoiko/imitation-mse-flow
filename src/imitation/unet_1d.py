"""
1D UNet implementation for action chunk flow policy.
Inspired by Diffusion Policy, see
https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/
"""
import math

import einops
import torch
import torch.nn as nn
from einops.layers.torch import Rearrange
from torch import Tensor


class Downsample1D(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Upsample1D(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x: Tensor) -> Tensor:
        return self.conv(x)


class Conv1DBlock(nn.Module):
    """Space dimension-preserving 1D convolutional block with GroupNorm and Mish."""

    def __init__(
        self, inp_channels: int, out_channels: int, kernel_size: int, n_groups: int = 8
    ) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(
                inp_channels, out_channels, kernel_size, padding=kernel_size // 2
            ),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal embedding layer for times in [0, 1]."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = torch.tensor(math.log(10000) / (half_dim - 1))
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :] * 2000
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class ConditionalResidualBlock1D(nn.Module):
    """Residual convolutional block with FiLM conditioning layer."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = False,
    ) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [
                Conv1DBlock(in_channels, out_channels, kernel_size, n_groups=n_groups),
                Conv1DBlock(out_channels, out_channels, kernel_size, n_groups=n_groups),
            ]
        )
        # FiLM modulation https://arxiv.org/abs/1709.07871
        # predicts per-channel scale and bias
        cond_channels = out_channels
        if cond_predict_scale:
            cond_channels = out_channels * 2
        self.cond_predict_scale = cond_predict_scale
        self.out_channels = out_channels
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, cond_channels),
            Rearrange("batch t -> batch t 1"),
        )
        # make sure dimensions compatible
        self.residual_conv = (
            nn.Conv1d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, cond: Tensor) -> Tensor:
        # cond ---------------\
        # x -> First conv -> FiLM -> Second conv -> + -> out
        #  \----------> residual conv -------------/
        out = self.blocks[0](x)
        embed = self.cond_encoder(cond)
        if self.cond_predict_scale:
            scale, bias = einops.rearrange(embed, "batch (n t) 1 -> n batch t 1", n=2)
            out = scale * out + bias
        else:
            out = out + embed
        out = self.blocks[1](out)
        out = out + self.residual_conv(x)
        return out


class ConditionalUnet1D(nn.Module):
    """1D UNet with conditioning."""

    def __init__(
        self,
        input_dim: int,
        cond_dim: int | None = None,
        flow_time_embed_dim: int = 256,
        down_dims: list[int] = [256, 512, 1024],
        kernel_size: int = 3,
        n_groups: int = 8,
        cond_predict_scale: bool = True,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)
        start_dim = down_dims[0]
        flow_time_encoder = nn.Sequential(
            SinusoidalPosEmb(flow_time_embed_dim),
            nn.Linear(flow_time_embed_dim, flow_time_embed_dim * 4),
            nn.Mish(),
            nn.Linear(flow_time_embed_dim * 4, flow_time_embed_dim),
        )
        cond_dim = (
            flow_time_embed_dim if cond_dim is None else cond_dim + flow_time_embed_dim
        )
        in_out = list(zip(all_dims[:-1], all_dims[1:]))
        down_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (len(in_out) - 1)
            down_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Downsample1D(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = all_dims[-1]
        self.mid_modules = nn.ModuleList(
            [
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
                ConditionalResidualBlock1D(
                    mid_dim,
                    mid_dim,
                    cond_dim=cond_dim,
                    kernel_size=kernel_size,
                    n_groups=n_groups,
                    cond_predict_scale=cond_predict_scale,
                ),
            ]
        )

        up_modules = nn.ModuleList([])
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (len(in_out) - 1)
            up_modules.append(
                nn.ModuleList(
                    [
                        ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim=cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                            cond_predict_scale=cond_predict_scale,
                        ),
                        Upsample1D(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        final_conv = nn.Sequential(
            Conv1DBlock(start_dim, start_dim, kernel_size=kernel_size),
            nn.Conv1d(start_dim, input_dim, 1),
        )
        self.flow_time_encoder = flow_time_encoder
        self.up_modules = up_modules
        self.down_modules = down_modules
        self.final_conv = final_conv

    def forward(
        self, sample: Tensor, time: Tensor, global_cond: Tensor | None = None
    ) -> Tensor:
        """
        x: (B,T,input_dim)
        timestep: (B,), time in [0, 1]
        local_cond: (B,T,local_cond_dim)
        global_cond: (B,global_cond_dim)
        output: (B,T,input_dim)
        """
        sample = einops.rearrange(sample, "b horizon t -> b t horizon")
        timesteps = time
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        global_feature = self.flow_time_encoder(timesteps)
        if global_cond is not None:
            global_feature = torch.cat([global_feature, global_cond], dim=-1)
        x = sample
        h = []
        for resnet, resnet2, downsample in self.down_modules:  # type: ignore
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            h.append(x)
            x = downsample(x)
        for mid_module in self.mid_modules:
            x = mid_module(x, global_feature)
        for resnet, resnet2, upsample in self.up_modules:  # type: ignore
            x, _ = einops.pack([x, h.pop()], "b * horizon")
            x = resnet(x, global_feature)
            x = resnet2(x, global_feature)
            x = upsample(x)
        x = self.final_conv(x)
        x = einops.rearrange(x, "b t h -> b h t")
        return x
