import types
from typing import List, Optional

import torch

from utils.scheduler import SchedulerInterface, FlowMatchScheduler
from wan.modules.tokenizers import HuggingfaceTokenizer
from wan.modules.action_skyreels_v2_i2v_model import ActionSkyReelsV2I2V
from wan.modules.causal_action_skyreels_v2_i2v_model import CausalActionSkyReelsV2I2V
from wan.modules.vae import _video_vae
from wan.modules.t5 import umt5_xxl


class WanTextEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.text_encoder = umt5_xxl(
            encoder_only=True,
            return_tokenizer=False,
            dtype=torch.float32,
            device=torch.device('cpu')
        ).eval().requires_grad_(False)
        self.text_encoder.load_state_dict(
            torch.load("checkpoints/SkyReels-V2-I2V-1.3B-540P/models_t5_umt5-xxl-enc-bf16.pth",
                       map_location='cpu', weights_only=False)
        )
        self.tokenizer = HuggingfaceTokenizer(
            name="checkpoints/SkyReels-V2-I2V-1.3B-540P/google/umt5-xxl/",
            seq_len=512, clean='whitespace')

    @property
    def device(self):
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(
            text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)
        for u, v in zip(context, seq_lens):
            u[v:] = 0.0
        return {"prompt_embeds": context}


class WanVAEWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        mean = [
            -0.7571, -0.7089, -0.9113, 0.1075, -0.1745, 0.9653, -0.1517, 1.5508,
            0.4134, -0.0715, 0.5517, -0.3632, -0.1922, -0.9497, 0.2503, -0.2921
        ]
        std = [
            2.8184, 1.4541, 2.3275, 2.6558, 1.2196, 1.7708, 2.6052, 2.0743,
            3.2687, 2.1526, 2.8652, 1.5579, 1.6382, 1.1253, 2.8251, 1.9160
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        self.model = _video_vae(
            pretrained_path="checkpoints/SkyReels-V2-I2V-1.3B-540P/Wan2.1_VAE.pth",
            z_dim=16,
        ).eval().requires_grad_(False)

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [B, C, F, H, W]
        pixel = pixel.contiguous()
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]
        output = [
            self.model.encode(u.unsqueeze(0), scale).float().squeeze(0)
            for u in pixel
        ]
        output = torch.stack(output, dim=0)
        # [B, C, F, H, W] -> [B, F, C, H, W]
        return output.permute(0, 2, 1, 3, 4)

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # latent: [B, F, C, H, W] -> [B, C, F, H, W]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype),
                 1.0 / self.std.to(device=device, dtype=dtype)]

        decode_function = self.model.cached_decode if use_cache else self.model.decode

        output = [decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0)
                  for u in zs]
        output = torch.stack(output, dim=0)
        return output.permute(0, 2, 1, 3, 4)


class ActionSkyReelsV2I2VWrapper(torch.nn.Module):
    """
    Wraps the action-conditioned SkyReels-V2 I2V backbone (bidirectional or
    causal) for flow-matching inference. Handles timestep reshaping, builds
    the [mask | image_latent] conditioning tensor, and converts the model's
    flow prediction to an x0 prediction.

    Inputs to forward():
      noisy_image_or_video: [B, F, 16, H', W']  WAN latent at timestep t
      conditional_dict:
        - "prompt_embeds": [B, L_txt, 4096] from WanTextEncoder
      timestep: [B, F] (bidirectional I2V uses 0 on frame 0) or [B]
      image_latent: [B, 16, F, H', W']  first-frame WAN-VAE latent
      clip_tokens:  [B, 257, 1280]      CLIP tokens for the first frame
      actions:      [B, F, r, D_action]
    """

    def __init__(
        self,
        model_name="SkyReels-V2-I2V-1.3B-540P",
        timestep_shift=5.0,
        is_causal=False,
        action_dim=10,
        local_attn_size=-1,
        sink_size=0,
    ):
        super().__init__()

        model_cls = CausalActionSkyReelsV2I2V if is_causal else ActionSkyReelsV2I2V
        self.model = model_cls.from_pretrained(
            f"checkpoints/{model_name}/",
            action_dim=action_dim,
            local_attn_size=local_attn_size,
            sink_size=sink_size,
            ignore_mismatched_sizes=True,
            low_cpu_mem_usage=False,
            device_map=None,
        )
        self.model.eval()

        assert getattr(self.model, "model_type", "i2v") == "i2v", \
            "WanModel must be in I2V mode."
        assert getattr(self.model, "in_dim", None) == 36, \
            "WanModel must have in_dim=36 for SkyReels I2V."

        # For non-causal diffusion, all frames share the same timestep.
        self.uniform_timestep = not is_causal

        self.scheduler = FlowMatchScheduler(
            shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        self.seq_len = 51000  # Matches a [1, 25, 16, 68, 120] latent grid
        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def _convert_flow_pred_to_x0(self, flow_pred: torch.Tensor, xt: torch.Tensor,
                                 timestep: torch.Tensor) -> torch.Tensor:
        """
        Flow matching: pred = noise - x0, x_t = (1 - sigma_t) * x0 + sigma_t * noise
        => x0 = x_t - sigma_t * pred
        """
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device),
            [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps])
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return (xt - sigma_t * flow_pred).to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(scheduler, x0_pred: torch.Tensor, xt: torch.Tensor,
                                 timestep: torch.Tensor) -> torch.Tensor:
        """Inverse of _convert_flow_pred_to_x0: pred = (x_t - x0) / sigma_t."""
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device),
            [x0_pred, xt, scheduler.sigmas, scheduler.timesteps])
        timestep_id = torch.argmin(
            (timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)
        sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        return ((xt - x0_pred) / sigma_t).to(original_dtype)

    def forward(
        self,
        noisy_image_or_video: torch.Tensor,
        conditional_dict: dict,
        timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        current_start: Optional[int] = None,
        cache_start: Optional[int] = None,
        image_latent: Optional[torch.Tensor] = None,
        clip_tokens: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        assert image_latent is not None, \
            "image_latent (first-frame WAN-VAE latent) is required."
        assert clip_tokens is not None, \
            "clip_tokens (CLIP tokens for the first frame) are required."

        prompt_embeds = conditional_dict["prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            # Bidirectional I2V uses t=0 on frame 0 and a shared t on the rest.
            if timestep.ndim == 2 and timestep.shape[1] > 1:
                first_zero = (timestep[:, 0] == 0).all().item()
                others_equal = (timestep[:, 1:] == timestep[:, 1:2]).all().item()
                is_i2v = first_zero and others_equal
                input_timestep = timestep[:, 1] if is_i2v else timestep[:, 0]
            else:
                input_timestep = timestep[:, 0]
        else:
            input_timestep = timestep

        B, F, C, H8, W8 = noisy_image_or_video.shape
        assert C == 16

        # Build the [mask(4) | image_latent(16)] conditioning tensor.
        # mask is 1 at frame 0 (reference) and 0 elsewhere.
        img_cond = image_latent
        mask = torch.ones_like(img_cond)
        mask[:, :, 1:] = 0.0
        y_cond = torch.cat([mask[:, :4], img_cond], dim=1)

        if kv_cache is not None:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                current_start=current_start,
                cache_start=cache_start,
                clip_fea=clip_tokens,
                y=y_cond,
                actions=actions,
            ).permute(0, 2, 1, 3, 4)
        else:
            flow_pred = self.model(
                noisy_image_or_video.permute(0, 2, 1, 3, 4),
                t=input_timestep,
                context=prompt_embeds,
                seq_len=self.seq_len,
                clip_fea=clip_tokens,
                y=y_cond,
                actions=actions,
            ).permute(0, 2, 1, 3, 4)

        pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=flow_pred.flatten(0, 1),
            xt=noisy_image_or_video.flatten(0, 1),
            timestep=timestep.flatten(0, 1),
        ).unflatten(0, flow_pred.shape[:2])

        return flow_pred, pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(
            SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(
            SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(
            SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        self.get_scheduler()

    def set_module_grad(self, module_grad: dict) -> None:
        """Enable or disable gradients on a set of named submodules."""
        for k, is_trainable in module_grad.items():
            getattr(self, k).requires_grad_(is_trainable)
