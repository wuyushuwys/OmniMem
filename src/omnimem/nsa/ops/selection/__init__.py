from .group.selection_two_pass import SelectionAttention2p as SelectionAttention2pGroup
from .token.selection_two_pass import SelectionAttention2p as SelectionAttention2pToken


def selection_attention(
        q, k, v,
        block_indices,
        block_count,
        block_size,
        chunk_size=1,
        scale=None,
        variant='two-pass',
        causal=False,
        return_lse=False,
        allow_tf32=True,
        bwd_method='auto',
        group_size=1,
):
    if variant not in ('one-pass', 'two-pass'):
        raise ValueError(f"Invalid variant: {variant!r}, expected 'one-pass' or 'two-pass'")
    if group_size < 1:
        raise ValueError(f"Invalid group_size: {group_size}, expected >= 1")

    if group_size > 1:
        if variant == 'one-pass':
            raise NotImplementedError(f"{variant=} is not supported")
        else:
            return SelectionAttention2pGroup.apply(
                q, k, v, block_indices, block_size, chunk_size,
                scale, causal, return_lse, allow_tf32, bwd_method, group_size,
            )
    else:
        if variant == 'one-pass':
            raise NotImplementedError(f"{variant=} is not supported")
        else:
            return SelectionAttention2pToken.apply(
                q, k, v, block_indices, block_size, chunk_size,
                scale, causal, return_lse, allow_tf32, bwd_method,
            )