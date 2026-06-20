import warnings
from functools import partial

import torch
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    ActivationWrapper,
    torch_utils_checkpoint,
    CheckpointImpl,
    _pack_kwargs,
    _unpack_kwargs,
)


class CheckpointWrapper(ActivationWrapper):
    """Wraps an nn.Module with activation checkpointing; use via checkpoint_wrapper()."""

    def __init__(
            self,
            mod: torch.nn.Module,
            checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
            checkpoint_fn=None,
            **checkpoint_fn_kwargs,
    ):
        super().__init__(mod)
        self.checkpoint_impl = checkpoint_impl
        if checkpoint_fn is None:
            # use torch.utils.checkpoint
            self.checkpoint_fn = partial(
                torch_utils_checkpoint,
                use_reentrant=(self.checkpoint_impl == CheckpointImpl.REENTRANT),
                **checkpoint_fn_kwargs,
            )
        else:
            # user-specified checkpoint function
            self.checkpoint_fn = partial(
                checkpoint_fn,
                **checkpoint_fn_kwargs,
            )

    def forward(self, *args, **kwargs):
        if not torch.is_grad_enabled():
            return self._checkpoint_wrapped_module(*args, **kwargs)
        if self.checkpoint_impl == CheckpointImpl.REENTRANT and kwargs != {}:
            flat_args, kwarg_keys = _pack_kwargs(*args, **kwargs)

            # Function that only takes (packed) args, but can unpack them
            # into the original args and kwargs for the checkpointed
            # function, and runs that function.
            def my_function(*inputs):
                unpacked_args, unpacked_kwargs = _unpack_kwargs(inputs, kwarg_keys)
                return self._checkpoint_wrapped_module(
                    *unpacked_args, **unpacked_kwargs
                )

            return self.checkpoint_fn(  # type: ignore[misc]
                my_function,
                *flat_args,
            )
        else:
            return self.checkpoint_fn(  # type: ignore[misc]
                self._checkpoint_wrapped_module, *args, **kwargs
            )


def checkpoint_wrapper(
        module: torch.nn.Module,
        checkpoint_impl: CheckpointImpl = CheckpointImpl.NO_REENTRANT,
        checkpoint_fn=None,
        **checkpoint_fn_kwargs,
) -> torch.nn.Module:
    """Wrap module for activation checkpointing.

    Args:
        module: Module to wrap.
        checkpoint_impl: Checkpointing implementation.
        checkpoint_fn: Custom checkpoint function; overrides default if set.
        **checkpoint_fn_kwargs: Passed to checkpoint_fn.

    Returns:
        Wrapped nn.Module.
    """

    if checkpoint_impl == CheckpointImpl.REENTRANT:
        warnings.warn(
            f"Please specify {CheckpointImpl.NO_REENTRANT} as "
            f"{CheckpointImpl.REENTRANT} will soon be removed as "
            "the default and eventually deprecated.",
            FutureWarning,
            stacklevel=2,
        )
    checkpoint_fn_kwargs["preserve_rng_state"] = False
    return CheckpointWrapper(
        module,
        checkpoint_impl,
        checkpoint_fn,
        **checkpoint_fn_kwargs,
    )
