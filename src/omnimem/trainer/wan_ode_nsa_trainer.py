import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler

from omnimem.models.transformers.causal_wan_nsa_model import CausalWanNSAModel, CausalWanNSAttentionBlock
from omnimem.utils.torch_utils import load_and_broadcast_diffuser
from omnimem.utils.meter import TimerMeter

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
        super().__init__(config)

    def build_model(self) -> CausalWanNSAModel:
        model: CausalWanNSAModel = load_and_broadcast_diffuser(
            model_cls=CausalWanNSAModel,
            model_name_or_path=self.config.model_path,
            device=self.device,
            low_cpu_mem_usage=False,
            **self.nsa_kwargs
        )
        model.init_gate()
        if dist.is_initialized():
            for name, p in model.named_parameters():
                if 'g_proj' in name:
                    dist.broadcast(p.data.to(self.device), src=0)

        if self.train_gate_only:
            for name, p in model.named_parameters():
                p.requires_grad_(('g_proj' in name))

        return model
