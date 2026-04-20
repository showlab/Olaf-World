from tqdm import tqdm
from typing import List
import torch

from wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from utils.action_skyreels_v2_i2v_wrapper import ActionSkyReelsV2I2VWrapper, WanTextEncoder, WanVAEWrapper

class ActionSkyVReelsI2VInferencePipeline(torch.nn.Module):
    def __init__(
            self,
            args,
            device,
            generator=None,
            text_encoder=None,
            vae=None,
            clip_model=None
    ):
        super().__init__()
        assert args.model_kwargs.is_causal == False, \
            "ActionSkyVReelsI2VInferencePipeline requires a bidirectional model."
        self.generator = ActionSkyReelsV2I2VWrapper(
            **getattr(args, "model_kwargs", {})) if generator is None else generator
        self.text_encoder = WanTextEncoder() if text_encoder is None else text_encoder
        self.vae = WanVAEWrapper() if vae is None else vae

        self.num_train_timesteps = args.num_train_timestep
        self.sampling_steps = 50
        self.sample_solver = 'unipc'
        self.shift = 5.0

        self.args = args

    @torch.no_grad()
    def inference(
        self,
        noise: torch.Tensor,
        text_prompts: List[str],
        first_latent: torch.Tensor = None, # torch.Size([1, 1, 16, 68, 120])
        init_latent: torch.Tensor = None, # torch.Size([1, 16, 25, 68, 120])
        clip_feats: torch.Tensor = None,
        actions: torch.Tensor = None,
        return_latents=False
    ) -> torch.Tensor:
        """Run classifier-free guided sampling.

        Args:
            noise: Initial noise, shape (B, T, C, H, W).
            text_prompts: Per-sample text prompts.

        Returns:
            Generated video in [0, 1], shape (B, T, C, H, W).
        """

        conditional_dict = self.text_encoder(
            text_prompts=text_prompts
        )
        unconditional_dict = self.text_encoder(
            text_prompts=[self.args.negative_prompt] * len(text_prompts)
        )

        latents = noise

        sample_scheduler = self._initialize_sample_scheduler(noise)
        for _, t in enumerate(tqdm(sample_scheduler.timesteps)):
            latent_model_input = latents
            timestep = t * torch.ones([latents.shape[0], latents.shape[1]], device=noise.device, dtype=torch.float32)

            flow_pred_cond, _ = self.generator(
                latent_model_input, conditional_dict, timestep,
                image_latent=init_latent, clip_tokens=clip_feats, actions=actions
            )
            flow_pred_uncond, _ = self.generator(
                latent_model_input, unconditional_dict, timestep,
                image_latent=init_latent, clip_tokens=clip_feats, actions=actions
            )

            flow_pred = flow_pred_uncond + self.args.guidance_scale * (
                flow_pred_cond - flow_pred_uncond)

            temp_x0 = sample_scheduler.step(
                flow_pred.unsqueeze(0),
                t,
                latents.unsqueeze(0),
                return_dict=False)[0]
            latents = temp_x0.squeeze(0)


        x0 = latents
        video = self.vae.decode_to_pixel(x0)
        video = (video * 0.5 + 0.5).clamp(0, 1)

        del sample_scheduler

        if return_latents:
            return video, latents
        else:
            return video

    def _initialize_sample_scheduler(self, noise):
        print(f"Initializing {self.sample_solver} sample scheduler (shift={self.shift})")
        if self.sample_solver == 'unipc':
            sample_scheduler = FlowUniPCMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sample_scheduler.set_timesteps(
                self.sampling_steps, device=noise.device, shift=self.shift)
            self.timesteps = sample_scheduler.timesteps
        elif self.sample_solver == 'dpm++':
            sample_scheduler = FlowDPMSolverMultistepScheduler(
                num_train_timesteps=self.num_train_timesteps,
                shift=1,
                use_dynamic_shifting=False)
            sampling_sigmas = get_sampling_sigmas(self.sampling_steps, self.shift)
            self.timesteps, _ = retrieve_timesteps(
                sample_scheduler,
                device=noise.device,
                sigmas=sampling_sigmas)
        else:
            raise NotImplementedError(f"Unsupported solver: {self.sample_solver}")
        return sample_scheduler
