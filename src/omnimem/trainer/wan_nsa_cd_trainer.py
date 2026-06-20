import torch
from torch.optim.lr_scheduler import LRScheduler


from omnimem.models.flex_attention_utils import create_causal_block_mask_cached
from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel, CausalWanNSAttentionBlock
from omnimem.pipelines import CausalWanT2VPipeline
from omnimem.schedulers import RectifiedFlowScheduler
from omnimem.utils.torch_utils import load_and_broadcast_diffuser
from omnimem.evaluate import evaluation_wan

from omnimem.trainer.wan_cd_trainer import WanConsistencyDistillTrainer


class WanConsistencyDistillNASTrainer(WanConsistencyDistillTrainer):
    model: CausalWanNSAModel
    ema: CausalWanNSAModel
    teacher: CausalWanNSAModel

    checkpoint_modules = CausalWanNSAttentionBlock

    def __init__(self, config):
        self.config = config
        self.window_size = self._get_and_record("window_size", None)
        self.sink_size = self._get_and_record("sink_size", None)
        self.nsa_kwargs = self._get_and_record("nsa_kwargs")
        super().__init__(config)
        self._evaluate_kwargs = dict(
            pipeline_cls=CausalWanT2VPipeline,
            frame_per_block=self.frame_per_block,
            sink_size=self.sink_size,
            window_size=self.window_size,
        )

    def build_model(self):
        model: CausalWanNSAModel = load_and_broadcast_diffuser(
            model_cls=CausalWanNSAModel,
            model_name_or_path=self.model_path,
            device=self.device,
            low_cpu_mem_usage=False,
            **self.nsa_kwargs,
        )
        self.logger.info(f"Load student {model.__class__.__name__} from {self.model_path}")

        return model

    def build_teacher(self):
        """Build the frozen teacher; teacher_path falls back to model_path."""
        teacher_path = self.teacher_path or self.model_path
        teacher: CausalWanNSAModel = load_and_broadcast_diffuser(
            model_cls=CausalWanNSAModel,
            model_name_or_path=teacher_path,
            device=self.device,
            low_cpu_mem_usage=False,
            **self.nsa_kwargs,
        )
        teacher.requires_grad_(False)
        teacher.eval()
        self.logger.info(f"Load teacher {teacher.__class__.__name__} from {teacher_path}")

        return teacher

    def _build_block_mask(self, block_size, sequence_length):
        return create_causal_block_mask_cached(
            block_size=block_size,
            B=None,
            H=None,
            Q_LEN=sequence_length,
            KV_LEN=sequence_length,
            use_flex_attention=True,
            torch_compile=True,
            window_chunks=self.window_size.get('kv') if self.window_size else None,
            sink_chunks=(self.sink_size.get('kv') if self.sink_size else None) or 0,
            teacher_forcing=self.teacher_forcing,
        )

    def evaluate(self):
        if self.ema is not None:
            evaluation_wan(
                transformer=self.ema,
                tokenizer=self.tokenizer,
                text_encoder=self.text_encoder,
                vae=self.vae,
                noise_scheduler=self.noise_scheduler,
                output_dir=self.output_dir,
                global_step=self.global_step,
                global_seed=self.global_seed,
                validation_data=self.validation_data,
                n_rows=self.validation_data.get("n_rows", 4),
                tag='ema/',
                timestep_shift=self.timestep_shift,
                device=self.device,
                s3_bucket=self.s3_bucket,
                s3_dir=self.s3_dir,
                upload=self.upload,
                **self._evaluate_kwargs,
            )
        evaluation_wan(
            transformer=self.model,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            vae=self.vae,
            noise_scheduler=self.noise_scheduler,
            output_dir=self.output_dir,
            global_step=self.global_step,
            global_seed=self.global_seed,
            validation_data=self.validation_data,
            n_rows=self.validation_data.get("n_rows", 4),
            timestep_shift=self.timestep_shift,
            device=self.device,
            s3_bucket=self.s3_bucket,
            s3_dir=self.s3_dir,
            upload=self.upload,
            **self._evaluate_kwargs,
        )
        torch.cuda.empty_cache()