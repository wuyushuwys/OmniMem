from typing import Union
from transformers import T5Tokenizer

from omnimem.models.autoencoders import AutoencoderKLWan
from omnimem.models.text_encoders import T5EncoderModel, HuggingfaceTokenizer
from omnimem.schedulers import RectifiedFlowScheduler

from omnimem.trainer.base_trainer import BaseTrainer


class BaseWanTrainer(BaseTrainer):
    text_encoder: T5EncoderModel
    tokenizer: Union[HuggingfaceTokenizer, T5Tokenizer]
    vae: AutoencoderKLWan
    noise_scheduler: RectifiedFlowScheduler

    def __init__(self, config):
        self.config = config
        self.text_encoder_path = self.config.text_encoder_path
        self.tokenizer_path = self.config.tokenizer_path
        self.vae_path = self.config.vae_path
        self.scheduler_path = self.config.scheduler_path
        self.wrap_text_encoder = config.get("wrap_text_encoder", True)

        super().__init__(config)

    def load_text_encoder(self):
        """Load T5 text encoder."""
        self.logger.info(f"Loading T5 from {self.text_encoder_path}")
        self.text_encoder = T5EncoderModel(
            text_len=512,
            device=self.device,
            dtype=self.dtype,
            checkpoint_path=self.text_encoder_path,
            tokenizer_path=self.tokenizer_path,
            fsdp_kwargs=self.fsdp_kwargs,
        )
        self.text_encoder.model.requires_grad_(False).eval().to(dtype=self.dtype)
        self.tokenizer = self.text_encoder.tokenizer

        if self.torch_compile:
            self.text_encoder.model.compile(dynamic=True, mode='max-autotune-no-cudagraphs')

    def load_vae(self):
        self.vae = AutoencoderKLWan.from_pretrained(self.vae_path).to(self.device)
        self.vae.requires_grad_(False).eval()

        self.logger.info(f"Load {self.vae.__class__.__name__} from {self.vae_path}")
        video_scale_factor = self.vae.temporal_downscale_factor
        vae_scale_factor = self.vae.spatial_downscale_factor
        self.logger.info(f"video_scale_factor: {video_scale_factor}\tvae_scale_factor: {vae_scale_factor}")

    def load_noise_scheduler(self):
        self.noise_scheduler = RectifiedFlowScheduler.from_config(
            self.scheduler_path,
        )
        self.logger.info(f"scheduler path {self.scheduler_path}")
        self.logger.info(f"{self.noise_scheduler.__class__.__name__} - {self.noise_scheduler.config.prediction_type}")

    def setup_frozen_components(self):
        self.load_text_encoder()
        self.load_vae()
        self.load_noise_scheduler()
