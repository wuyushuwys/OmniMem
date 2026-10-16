"""The streaming request :class:`Recipe` and its defaults; torch-free, so the CLI builds its parser before the NUMA binding."""
from dataclasses import dataclass
from typing import Optional

# Scratch default covers the largest working set of one selection call at 480x832.
DEFAULT_BLOCK_POOL_GB = 24.0
DEFAULT_BLOCK_SCRATCH_GB = 0.75


@dataclass
class Recipe:
    height: int = 480
    width: int = 832
    fps: int = 16
    num_inference_steps: int = 4
    shift: float = 5.0
    frame_per_block: int = 3
    window_size: Optional[int] = 3
    sink_size: Optional[int] = None
    enable_kv_evict: bool = True
    # Chunk LRU capacity of 'chunk_lru'; residency only.
    lru_max_size: int = 20
    compile: bool = True
    # "default": max-autotune picks Inductor kernels by timing, which is not reproducible.
    compile_mode: str = "default"
    # Opt-in: reshape + F.linear patch embedding is faster but may not round like Conv3d.
    patch_embed_linear: bool = False
    crf: int = 18
    preset: str = "medium"
    pix_fmt: str = "yuv420p"
    slice_last_frames: int = 21
    recache_mode: str = "full"
    park_t5: Optional[str] = None  # e.g. 'cuda:1' or 'cpu': park the text encoder after encoding
    # KV history policy: 'block_lru' (needs a finite KV window and enable_kv_evict) or 'chunk_lru'.
    kv_cache_policy: str = "block_lru"
    # Persistent block pool budget, 'block_lru' only; None uses DEFAULT_BLOCK_POOL_GB.
    kv_block_pool_gb: Optional[float] = None
    # Shared scratch pool budget, 'block_lru' only; None uses DEFAULT_BLOCK_SCRATCH_GB.
    kv_block_scratch_gb: Optional[float] = None
    # Block store options, applied only under 'block_lru'; resident runs the whole KV on GPU.
    kv_block_resident_if_fits: bool = True
    # Block key dedup: 'device' kernel (default, bit-exact) or torch.unique.
    kv_block_dedup: str = "device"
    # GPU history tier, opt-in: evict KV history to the decode GPU instead of
    # pinned host memory.  Needs the two-GPU layout and ``kv_cache_policy='chunk_lru'``.
    kv_tier: bool = False
    # Budget of the tier on the decode GPU, in GiB; evictions past it fall back to pinned
    # host memory.  A 960-frame request evicts 54.5 GiB with the default chunk LRU.
    kv_tier_budget_gb: float = 60.0
    # Contiguous history / window buffers instead of per-step torch.cat; faster, uses more memory.
    kv_window_buffer: bool = True
    # Fast VAE decode: channels-last convs, one temporal pass per chunk; pixels differ slightly.
    vae_decode_fast: bool = True
