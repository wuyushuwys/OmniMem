import random
from typing import Optional

import torch
import torch.distributed as dist

from omnimem.trainer.wan_nsa_self_forcing_streaming_trainer import WanNSASelfForcingStreamingTrainer


class WanNSASelfForcingStreamingSwitchTrainer(WanNSASelfForcingStreamingTrainer):
    """
    Extends the streaming trainer with mid-sequence prompt switching.

    Config keys: switch_prompt_file (str), switch_mode (str), switch_choices (list[int]).
    Falls back to base behavior when switch_prompt_file is not set.
    """

    def __init__(self, config):
        self.switch_prompt_file = config.get("switch_prompt_file", None)
        self.switch_mode = config.get("switch_mode", "random_choice")
        self.switch_choices = list(config.get("switch_choices", []))

        super().__init__(config)

        if self.switch_prompt_file is not None:
            with open(self.switch_prompt_file, "r") as f:
                self._switch_prompts = [l.strip() for l in f if l.strip()]
        else:
            self._switch_prompts = []

    def _get_switch_frame_index(self, max_length: int) -> int:
        """Pick a switch frame from switch_choices, synced across all ranks."""
        valid = [c for c in self.switch_choices if 0 < c < max_length]
        if not valid:
            return max_length  # no valid choice → no switch

        idx = torch.zeros(1, dtype=torch.long, device=self.device)
        if not dist.is_initialized() or dist.get_rank() == 0:
            idx.fill_(random.choice(valid))
        if dist.is_initialized():
            dist.broadcast(idx, src=0)
        return idx.item()

    def start_new_sequence(self):
        """Initialize a new sequence via parent, then append switch prompt and frame index to streaming_state."""
        super().start_new_sequence()

        max_length: int = self.streaming_state["temp_max_length"]
        sequence_length: int = self.streaming_state["condition_dict"]["seq_len"]
        batch_size: int = len(self.streaming_state["condition_dict"]["context"])

        switch_frame_index: int = max_length      # default: no switch
        switch_condition_dict: Optional[dict] = None

        if self._switch_prompts:
            sel = torch.zeros(1, dtype=torch.long, device=self.device)
            if not dist.is_initialized() or dist.get_rank() == 0:
                sel.fill_(random.randint(0, len(self._switch_prompts) - 1))
            if dist.is_initialized():
                dist.broadcast(sel, src=0)

            switch_text = [self._switch_prompts[sel.item()]] * batch_size
            switch_embeddings, _ = self.text_encoder(
                texts=switch_text, device=self.device
            )
            switch_condition_dict = dict(
                context=switch_embeddings, seq_len=sequence_length
            )

            switch_frame_index = self._get_switch_frame_index(max_length)

        self.streaming_state["switch_frame_index"] = switch_frame_index
        self.streaming_state["switch_condition_dict"] = switch_condition_dict

    @torch.no_grad()
    def _recache_after_switch(self):
        """Re-run a context-pass over recent latents with the new prompt, overwriting their self-attn KV."""
        state = self.streaming_state
        prev = state["previous_frames"]
        if prev is None or self.streaming_kv_cache is None:
            return
        n = prev.shape[2]
        start = state["current_length"] - n
        if start < 0:
            return
        fpb = self.frame_per_block
        batch_size = len(state["condition_dict"]["context"])
        context_t = torch.zeros(batch_size, device=prev.device, dtype=prev.dtype)
        for i in range(0, n, fpb):
            self.generator(
                x=prev[:, :, i:i + fpb].contiguous(),
                t=context_t,
                **state["condition_dict"],
                start_frame=start + i,
                kv_cache=self.streaming_kv_cache,
            )
        if self.enable_kv_evict:
            self.streaming_kv_cache.flush_offload()

    def generate_next_chunk(self, requires_grad: bool = True):
        """Switch to the second prompt and recache recent KV when current_length reaches switch_frame_index."""
        state = self.streaming_state
        switch_cond = state.get("switch_condition_dict")
        switch_frame = state.get("switch_frame_index")

        if (switch_cond is not None and state["current_length"] >= switch_frame
                and self.streaming_active and self.can_generate_more()):
            self.logger.info(
                f"[switch] frame {state['current_length']} >= {switch_frame}, "
                "switching to second prompt"
            )
            state["condition_dict"] = switch_cond
            state["switch_condition_dict"] = None
            self._recache_after_switch()

        return super().generate_next_chunk(requires_grad)
