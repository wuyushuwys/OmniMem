from functools import lru_cache
import math
from torch.nn.attention.flex_attention import (
    create_block_mask,
    create_mask,
)


@lru_cache
def create_causal_block_mask_cached(
        block_size, B, H,
        Q_LEN, KV_LEN, device="cuda",
        use_flex_attention=False,
        torch_compile=False,
        padded_length=None,
        kv_stride=1,
        window_chunks=None,
        sink_chunks=0,
        teacher_forcing=False,
):
    """
    Build a (cached) flex_attention BlockMask for causal Wan training.

    When teacher_forcing=True, the sequence is [clean | noisy] (length 2*Q_LEN):
      clean->clean: block-causal; clean->noisy: forbidden;
      noisy->clean: strict causal (chunk i sees clean 0..i-1 only);
      noisy->noisy: same chunk only (no cross-chunk noisy history at inference).
    Aligned with thu-ml/Causal-Forcing `_prepare_teacher_forcing_mask`.
    """
    use_window = window_chunks is not None
    if use_window:
        assert sink_chunks >= 0
        assert window_chunks >= 0

    if teacher_forcing:
        q_half_chunks = Q_LEN // block_size
        kv_half_chunks = (KV_LEN * kv_stride) // block_size
        Q_LEN = Q_LEN * 2
        KV_LEN = KV_LEN * 2

    if padded_length is not None:
        Q_LEN = math.ceil(Q_LEN / padded_length) * padded_length
        KV_LEN = math.ceil(KV_LEN / padded_length) * padded_length

    def causal_block_mask(b, h, q_idx, kv_idx):
        q_chunk = q_idx // block_size
        kv_chunk = (kv_idx * kv_stride) // block_size

        if teacher_forcing:
            q_in_teacher = q_chunk < q_half_chunks
            kv_in_teacher = kv_chunk < kv_half_chunks

            # Map chunk indices to their position within their own half [0, N).
            q_rel  = q_chunk  - (~q_in_teacher)  * q_half_chunks
            kv_rel = kv_chunk - (~kv_in_teacher) * kv_half_chunks

            # Case 1: clean q -> clean kv  : block-causal (kv_rel <= q_rel)
            clean_to_clean = q_in_teacher & kv_in_teacher & (kv_rel <= q_rel)

            # Case 2: clean q -> noisy kv  : forbidden (omitted from clauses)

            # Case 3: noisy q -> clean kv  : strict causal (kv_rel < q_rel)
            #   noisy chunk i sees clean chunks 0..i-1 only, NOT clean chunk i
            noisy_to_clean = (~q_in_teacher) & kv_in_teacher & (kv_rel < q_rel)

            # Case 4: noisy q -> noisy kv  : same chunk only (kv_rel == q_rel)
            #   noisy chunk i only sees noisy tokens within its own chunk
            noisy_to_noisy = (~q_in_teacher) & (~kv_in_teacher) & (kv_rel == q_rel)

            if use_window:
                # Window only affects the cross-chunk paths (clean->clean,
                # noisy->clean). Same-chunk noisy->noisy is always allowed.
                is_sink = kv_rel < sink_chunks
                is_local = (q_rel - kv_rel) < window_chunks
                window_ok = is_sink | is_local
                mask = (clean_to_clean & window_ok) | (noisy_to_clean & window_ok) | noisy_to_noisy
            else:
                mask = clean_to_clean | noisy_to_clean | noisy_to_noisy
        else:
            causal = q_chunk >= kv_chunk
            if use_window:
                is_sink = kv_chunk < sink_chunks
                is_local = (q_chunk - kv_chunk) < window_chunks
                causal = causal & (is_sink | is_local)
            mask = causal

        return mask

    if use_flex_attention:
        return create_block_mask(
            causal_block_mask,
            None,
            None,
            Q_LEN,
            KV_LEN,
            device=device,
            _compile=torch_compile
        )
    else:
        return create_mask(causal_block_mask, None, None, Q_LEN, KV_LEN, device=device)