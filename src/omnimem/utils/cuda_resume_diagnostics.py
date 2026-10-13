"""Opt-in CUDA phase synchronization for the first resumed steps, to locate async errors."""

import functools
import os
import types

import torch


_DIAGNOSTIC_STEPS_ENV = "OMNIMEM_CUDA_RESUME_DIAGNOSTIC_STEPS"
_PHASE_NAMES = frozenset(("self_attn", "cross_attn", "ffn"))


def enable_cuda_resume_diagnostics(trainer):
    """Instrument the next N ``train_step`` calls, N from OMNIMEM_CUDA_RESUME_DIAGNOSTIC_STEPS."""
    raw_steps = os.environ.get(_DIAGNOSTIC_STEPS_ENV, "0")
    try:
        diagnostic_steps = int(raw_steps)
    except ValueError as error:
        raise ValueError(f"{_DIAGNOSTIC_STEPS_ENV} must be a nonnegative integer, got {raw_steps!r}") from error
    if diagnostic_steps < 0:
        raise ValueError(f"{_DIAGNOSTIC_STEPS_ENV} must be a nonnegative integer, got {raw_steps!r}")
    if diagnostic_steps == 0:
        return

    original_train_step = trainer.train_step
    successful_calls = 0

    @functools.wraps(original_train_step)
    def diagnostic_train_step(self, *args, **kwargs):
        nonlocal successful_calls
        handles = []
        step = self.global_step
        diagnostic_call = successful_calls + 1
        succeeded = False

        def phase_hook(phase_name):
            def synchronize_phase(module, inputs, output):
                try:
                    torch.cuda.synchronize(self.device)
                except Exception:
                    self.logger.exception(
                        "CUDA resume diagnostic synchronization failed: "
                        "global_step=%s diagnostic_call=%s/%s generator_phase=%s",
                        step, diagnostic_call, diagnostic_steps, phase_name,
                    )
                    raise
            return synchronize_phase

        self.logger.info(
            "CUDA resume diagnostics start: global_step=%s diagnostic_call=%s/%s",
            step, diagnostic_call, diagnostic_steps,
        )
        try:
            for name, module in self.generator.named_modules():
                if name.rsplit(".", 1)[-1] in _PHASE_NAMES:
                    handles.append(module.register_forward_hook(phase_hook(name)))
            result = original_train_step(*args, **kwargs)
            succeeded = True
            successful_calls += 1
            return result
        finally:
            for handle in handles:
                handle.remove()
            if successful_calls >= diagnostic_steps:
                self.train_step = original_train_step
            self.logger.info(
                "CUDA resume diagnostics end: global_step=%s diagnostic_call=%s/%s status=%s",
                step, diagnostic_call, diagnostic_steps,
                "success" if succeeded else "failed",
            )

    trainer.train_step = types.MethodType(diagnostic_train_step, trainer)
