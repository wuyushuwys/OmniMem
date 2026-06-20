from typing import Optional, Union, Tuple
from dataclasses import dataclass

import numpy as np
import math

import torch
from torch import Tensor
from torch.distributions import LogisticNormal

from diffusers.utils import BaseOutput
from diffusers.configuration_utils import register_to_config, ConfigMixin
from diffusers.schedulers.scheduling_utils import SchedulerMixin


@dataclass
class RectifiedFlowSchedulerOutput(BaseOutput):
    """
    Output class for the schedulers's `step` function output.

    Args:
        prev_sample (`torch.FloatTensor` of shape `(batch_size, num_channels, height, width)` for images):
            Computed sample `(x_{t-1})` of previous timestep. `prev_sample` should be used as next model input in the
            denoising loop.
    """

    prev_sample: Union[torch.FloatTensor, torch.Tensor]


def calculate_shift(
        image_seq_len,
        base_seq_len: int = 144,
        max_seq_len: int = 4096,
        base_shift: float = 0.95,
        max_shift: float = 2.05,
):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu


def _time_shift_exponential(mu, sigma, t):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


class RectifiedFlowScheduler(SchedulerMixin, ConfigMixin):
    """Rectified flow scheduler for unified training.

    Args:
        num_train_timesteps: Number of diffusion training steps.
        shift: Shift value for the timestep schedule.
    """
    _compatibles = []
    order = 1
    init_noise_sigma = 1.
    _step_index = None
    sequence_length_per_frame = None

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            shift: Optional[float] = 1.0,
            sample_method: str = 'logit-normal',
            loc: float = 0., scale: float = 1.,
            eps: float = 1e-3,
            init_noise_sigma: float = 1.,
            base_spatial: int = 144,
            base_temporal: int = 1,
            max_shift: Optional[int] = None,
            disable_timestep_transform: Optional[bool] = False,
            shift_transform: Optional[bool] = False,
            time_shift_type: Optional[str] = "linear",
            discrete_timesteps: Optional[bool] = False,
            causal: Optional[bool] = False,
            frame_per_block: Optional[int] = None,
            prediction_type='v_prediction',
            sigma_max: float = 1.0
    ):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.sample_method = sample_method
        self.init_noise_sigma = init_noise_sigma
        self.eps = eps
        self.shift = shift
        self.base_spatial = base_spatial
        self.base_temporal = base_temporal
        self.base_seq_len = base_spatial * base_temporal
        self.prediction_type = prediction_type
        self.max_shift = max_shift
        self.disable_timestep_transform = disable_timestep_transform

        self.logit_loc = loc
        self.logit_scale = scale
        self.shift_transform = shift_transform

        self.discrete_timesteps = discrete_timesteps
        self.causal = causal
        self.frame_per_block = frame_per_block or 1

        alphas = np.linspace(sigma_max - self.eps, 0, num_train_timesteps)[::-1].copy()

        sigmas = torch.from_numpy(1.0 - alphas).to(dtype=torch.float32)

        if shift != 1:
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # pyright: ignore

        self.sigmas = sigmas
        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

        self.timesteps = sigmas * num_train_timesteps

        if sample_method == 'logit-normal':
            self._distribution = LogisticNormal(
                loc=torch.tensor([self.logit_loc]),
                scale=torch.tensor([self.logit_scale])
            )
        elif sample_method == 'uniform':
            pass
        else:
            raise NotImplementedError(f"{sample_method} not implemented")

    def scale_model_input(self, sample: torch.Tensor, timestep: Optional[int] = None) -> torch.Tensor:
        """No-op: returns sample unchanged (compatibility with other schedulers)."""
        return sample

    def scale_noise(self, noise: torch.Tensor) -> torch.Tensor:
        return noise * self.init_noise_sigma

    def _sample_from_dist(self, batch_size, device, sample_method=None):
        sample_method = sample_method or self.sample_method
        if sample_method == 'logit-normal':
            return self._distribution.sample((batch_size,))[:, 0].to(
                dtype=torch.float32, device=device)
        elif sample_method == 'uniform':
            return torch.rand((batch_size,), device=device)
        else:
            raise NotImplementedError

    @torch.compiler.disable()
    @torch.no_grad()
    def sample_timesteps(
            self,
            batch_size: int,
            noise: Optional[torch.Tensor] = None,
            device: Union[str, torch.device] = None,
            sample_method: Optional[str] = None,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:

        sample_method = sample_method or self.sample_method

        timesteps = self._sample_from_dist(batch_size, device, sample_method=sample_method)

        timesteps = timesteps.clamp(self.sigma_min, self.sigma_max) * self.num_train_timesteps

        if self.discrete_timesteps:
            timesteps = timesteps.to(torch.long)
        return timesteps

    def shift_timesteps(self, timesteps: torch.Tensor, shift: float = None) -> torch.Tensor:
        sigmas = timesteps / self.config.num_train_timesteps
        sigmas.clip_(self.sigma_min, self.sigma_max)
        if shift is None:
            shift = self.config.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)  # pyright: ignore
        return sigmas * self.config.num_train_timesteps

    @torch.compiler.disable()
    def scale_timesteps(self, timesteps: torch.Tensor, noise=None, overwrite_ratio=None) -> torch.Tensor:
        """Scale timesteps by normalizing then applying timestep_transform."""
        timepoints = timesteps / self.num_train_timesteps
        timepoints = self.timestep_transform(timepoints, noise=noise, overwrite_ratio=overwrite_ratio)
        return timepoints * self.num_train_timesteps

    def add_noise(
            self,
            original_samples: torch.Tensor,
            noise: torch.Tensor,
            timesteps: torch.Tensor,
    ) -> torch.FloatTensor:

        timepoints = timesteps.to(original_samples) / self.num_train_timesteps

        timepoints = 1 - timepoints

        bsz = noise.shape[0]

        if timesteps.ndim == 1:
            timepoints = timepoints.reshape((bsz,) + (1,) * (noise.ndim - 1))
        elif timesteps.ndim == 2:
            if noise.ndim == 3:
                # noise: [batch_size, sequence_length, num_channel]
                timepoints = timepoints.reshape(bsz, -1, 1)
            elif noise.ndim == 5:
                # noise: [batch_size, num_channel, num_frames, height, width]
                batch_size, _, num_frames, height, width = noise.shape
                timepoints = timepoints.reshape(batch_size, 1, num_frames, height, width)
            else:
                raise NotImplementedError(f"shape not support {noise.shape}")
        else:
            raise ValueError(f"timesteps.ndim {timesteps.ndim} != noise.ndim {noise.ndim}")

        return (1 - timepoints) * noise + timepoints * original_samples

    def timestep_transform(self, t, noise=None, overwrite_ratio=None):
        """Apply flow-matching shift to normalized timesteps.

        Args:
            t: sampled timesteps in [0, 1].
            noise: noise tensor used to infer spatial/temporal dimensions for adaptive shift.
            overwrite_ratio: explicit shift ratio, overrides config shift.
        """
        if overwrite_ratio is not None:
            shift = overwrite_ratio
            return shift * t / (1 + (shift - 1) * t)
        if self.disable_timestep_transform:
            return t
        if self.shift != 1:
            return self.shift * t / (1 + (self.shift - 1) * t)
        if noise is not None:
            if self.shift_transform:
                image_len = np.prod(noise.shape[3:])
                mu = calculate_shift(image_len)
                if self.config.time_shift_type == "exponential":
                    t = _time_shift_exponential(mu, 1.0, t)
                elif self.config.time_shift_type == 'linear':
                    shift = math.sqrt(image_len / self.base_spatial)
                    t = shift * t / (1 + (shift - 1) * t)
            else:
                seq_len = np.prod(noise.shape[2:])
                shift = math.sqrt(seq_len / self.base_seq_len)
                t = shift * t / (1 + (shift - 1) * t)

        return t

    @torch.no_grad()
    def get_loss_weight(self, noise=None):
        """Compute per-sample loss weight based on noise spatial/temporal dimensions.

        Args:
            noise: noise tensor used to infer dimensions.
        """
        if noise is not None:
            if noise.ndim == 5:
                dim_t, dim_h, dim_w = noise.shape[2:]
            elif noise.ndim == 4:
                dim_h, dim_w = noise.shape[2:]
                dim_t = 1
            else:
                raise NotImplementedError(f"shape not support {noise.shape}")
            ratio_spatial = np.sqrt((dim_h * dim_w) / self.base_spatial)
            ratio_temporal = np.sqrt(dim_t / self.base_temporal)
            ratio = ratio_spatial * ratio_temporal
        else:
            ratio = 1
        return 1 / ratio

    @torch.compiler.disable()
    def get_velocity(self, sample: torch.Tensor, noise: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        velocity = noise - sample
        return velocity

    @torch.compiler.disable()
    def set_timesteps(
            self,
            num_inference_steps: int,
            device: Union[str, torch.device] = None,
            noise: Optional[torch.Tensor] = None,
            overwrite_ratio: Optional[float] = None,
            sequence_length_per_frame: Optional[int] = None,
    ) -> None:
        """Set discrete inference timesteps from sigma=1.0 down to sigma=0.0.

        Args:
            num_inference_steps: Number of denoising steps.
            device: Target device for timestep tensors.
            noise: Noise tensor used to determine adaptive shift.
            overwrite_ratio: Explicit shift ratio to override config.
            sequence_length_per_frame: Sequence length per frame for causal inference.
        """
        self.num_inference_steps = num_inference_steps
        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1).copy()[:-1]  # pyright: ignore
        sigmas = torch.from_numpy(sigmas).to(device=device, dtype=torch.float32)
        sigmas = self.timestep_transform(sigmas, noise=noise, overwrite_ratio=overwrite_ratio)
        timesteps = sigmas * self.config.num_train_timesteps
    
        self.sigmas = torch.cat([sigmas, torch.zeros(1, device=sigmas.device)])
        self.timesteps = timesteps.to(device=device, dtype=torch.int64)
        self._step_index = 0
        self.sequence_length_per_frame = sequence_length_per_frame


    @torch.compiler.disable()
    def step(
            self,
            model_output: torch.FloatTensor,
            timestep: Optional[Union[float, torch.FloatTensor]],
            sample: torch.FloatTensor,
            return_dict: bool = True,
    ) -> Union[RectifiedFlowSchedulerOutput, Tuple]:
        dtype = model_output.dtype
        sample = sample.to(torch.float32)
        model_output = model_output.to(torch.float32)
        dt = self.sigmas[self.step_index + 1] - self.sigmas[self.step_index]

        prev_sample = sample + model_output * dt.expand((model_output.shape[0],) + (1,) * (model_output.ndim - 1))
        prev_sample = prev_sample.to(dtype)

        self._step_index += 1

        if not return_dict:
            return (prev_sample,)

        return RectifiedFlowSchedulerOutput(prev_sample=prev_sample)

    @property
    def step_index(self) -> int:
        return self._step_index % self.num_inference_steps