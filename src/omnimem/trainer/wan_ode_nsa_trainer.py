import torch
from torch.optim.lr_scheduler import LRScheduler

from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel, CausalWanNSAttentionBlock
from omnimem.utils.torch_utils import load_and_broadcast_diffuser
from omnimem.utils.meter import TimerMeter
from omnimem.utils.misc import init_gate_on_all_ranks

from omnimem.trainer.wan_ode_trainer import WanODETrainer


class WanNSAODETrainer(WanODETrainer):
    model: CausalWanNSAModel

    optimizer: torch.optim.Optimizer
    lr_scheduler: LRScheduler

    forward_timer: TimerMeter
    backward_timer: TimerMeter

    checkpoint_modules = CausalWanNSAttentionBlock

    def __init__(self, config):
        self.config = config
        self.nsa_kwargs = self._get_and_record("nsa_kwargs")
        self.train_gate_only = self._get_and_record("train_gate_only")
        self.preserve_pretrained_gate = self._get_and_record("preserve_pretrained_gate", False)
        super().__init__(config)

    def build_model(self) -> CausalWanNSAModel:
        model: CausalWanNSAModel = load_and_broadcast_diffuser(
            model_cls=CausalWanNSAModel,
            model_name_or_path=self.config.model_path,
            device=self.device,
            low_cpu_mem_usage=False,
            **self.nsa_kwargs
        )
        # NSA checkpoints already contain trained gates. Keep them when finetuning.
        if not self.preserve_pretrained_gate:
            init_gate_on_all_ranks(model, self.device)

        if self.train_gate_only:
            for name, p in model.named_parameters():
                p.requires_grad_(('g_proj' in name))

        return model
