"""Native C++/CUDA backend for the OmniMem NSA operators, built lazily on first use."""
from ._loader import is_loaded, load, library_path  # noqa: F401
