"""Build and load one JIT-compiled CUDA extension, shared by the NSA and block-KV loaders."""
from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import importlib.util
import logging
import os
import shutil
import socket
import sysconfig
import threading
import time
from pathlib import Path
from typing import Any, Callable, Iterator, List, Optional, Tuple, Type

import torch

from omnimem.nsa.backend import NSABackendError

CSRC_DIR = Path(__file__).resolve().parent / "csrc"
CFLAGS = ("-O3", "-std=c++17")

#: Seconds to wait for another process that is building the same extension.
BUILD_LOCK_TIMEOUT_S = 1800.0
_LOCK_POLL_S = 0.5
_LOCK_REPORT_S = 60.0

logger = logging.getLogger(__name__)


@dataclasses.dataclass(frozen=True)
class Extension:
    """One extension: its sources (file names under ``csrc/``) and how to build it."""

    prefix: str                    # library name without the build key
    what: str                      # for messages, e.g. "NSA CUDA extension"
    sources: Tuple[str, ...]
    headers: Tuple[str, ...]
    cuda_cflags: Tuple[str, ...]
    python_module: bool            # a pybind11 module (True) or a torch.ops library (False)
    error: Type[NSABackendError]

    def source_files(self) -> List[str]:
        return [str(CSRC_DIR / n) for n in self.sources]

    def arch_list(self) -> str:
        """The ``-gencode`` targets: ``TORCH_CUDA_ARCH_LIST``, else the visible GPUs."""
        env = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if env:
            return env
        if not torch.cuda.is_available():
            raise self.error(
                f"the {self.what} requires a CUDA device (or TORCH_CUDA_ARCH_LIST for an "
                "explicit cross-compile target).")
        caps = sorted({torch.cuda.get_device_capability(i)
                       for i in range(torch.cuda.device_count())})
        return ";".join(f"{a}.{b}" for a, b in caps)

    def check_toolchain(self) -> str:
        """Raise unless nvcc and ninja are available; returns ``CUDA_HOME``."""
        from torch.utils import cpp_extension

        cuda_home = cpp_extension.CUDA_HOME
        if cuda_home is None:
            raise self.error(
                f"the {self.what} has to be built but no CUDA toolkit was found (CUDA_HOME "
                "unset and nvcc not on PATH). Install a CUDA toolkit matching "
                f"torch.version.cuda={torch.version.cuda} or use the Triton backend.")
        nvcc = os.path.join(cuda_home, "bin", "nvcc")
        if not os.path.exists(nvcc) and shutil.which("nvcc") is None:
            raise self.error(f"nvcc not found under CUDA_HOME={cuda_home}")
        if shutil.which("ninja") is None:
            raise self.error(f"ninja is required to build the {self.what} (pip install ninja)")
        return cuda_home

    def build_key(self, arch_list: str) -> str:
        """Hash of everything the binary depends on (see the module docstring)."""
        h = hashlib.sha256()

        def feed(data: bytes) -> None:
            h.update(len(data).to_bytes(8, "little"))
            h.update(data)

        for name in self.sources + self.headers:
            feed(name.encode())
            feed((CSRC_DIR / name).read_bytes())
        for part in (*CFLAGS, *self.cuda_cflags, torch.__version__, str(torch.version.cuda),
                     str(torch._C._GLIBCXX_USE_CXX11_ABI), arch_list):
            feed(part.encode())
        if self.python_module:
            feed((sysconfig.get_config_var("SOABI") or "").encode())
        return h.hexdigest()[:12]

    def name(self, arch_list: Optional[str] = None) -> str:
        return f"{self.prefix}_{self.build_key(arch_list or self.arch_list())}"

    def build_and_load(self, verbose: bool = False) -> Tuple[Optional[Any], str]:
        """Build the extension unless this exact build exists, then load it."""
        from torch.utils import cpp_extension

        arch_list = self.arch_list()
        name = self.name(arch_list)
        root = os.environ.get("TORCH_EXTENSIONS_DIR") or cpp_extension.get_default_build_root()
        build_dir = os.path.join(root, name)
        library = os.path.join(build_dir, f"{name}.so")
        stamp = os.path.join(build_dir, "build.ok")

        def is_built() -> bool:
            return os.path.exists(stamp) and os.path.exists(library)

        module = None
        if not is_built():
            self.check_toolchain()
            os.makedirs(build_dir, exist_ok=True)
            with self._build_lock(build_dir):
                if not is_built():          # nobody finished it while this process waited
                    module = self._build(name, arch_list, build_dir, verbose)
                    Path(stamp).touch()
        if not self.python_module:
            torch.ops.load_library(library)
        elif module is None:
            module = _import_module(name, library)
        return module, library

    def _build(self, name: str, arch_list: str, build_dir: str, verbose: bool) -> Optional[Any]:
        """Compile, link and load; called with the build lock held."""
        from torch.utils import cpp_extension

        # torch's JIT keeps a lock file of its own and waits on it without a limit.  Builds
        # here are already serialised, so one found now was left behind by a killed build.
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(build_dir, "lock"))
        logger.info("building the %s in %s", self.what, build_dir)
        with _environ("TORCH_CUDA_ARCH_LIST", arch_list):      # read by cpp_extension
            loaded = cpp_extension.load(
                name=name,
                sources=self.source_files(),
                extra_cflags=list(CFLAGS),
                extra_cuda_cflags=list(self.cuda_cflags),
                extra_include_paths=[str(CSRC_DIR)],
                build_directory=build_dir,
                verbose=verbose,
                with_cuda=True,
                is_python_module=self.python_module,
                keep_intermediates=True,
            )
        return loaded if self.python_module else None

    @contextlib.contextmanager
    def _build_lock(self, build_dir: str) -> Iterator[None]:
        """Hold ``<build_dir>/build.lock`` (created exclusively, removed on exit)."""
        path = os.path.join(build_dir, "build.lock")
        start = time.monotonic()
        next_report = start
        while True:
            try:
                fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
                break
            except FileExistsError:
                pass
            now = time.monotonic()
            hint = (f"build lock {path} (held by {_read_text(path) or 'an unknown process'}). "
                    "If no build of it is running any more, a killed build left the file "
                    "behind: delete it")
            if now - start >= BUILD_LOCK_TIMEOUT_S:
                raise self.error(
                    f"gave up after {BUILD_LOCK_TIMEOUT_S:.0f} s waiting for the {self.what} "
                    f"{hint} and retry.")
            if now >= next_report:
                logger.warning("waiting for another process to build the %s: %s. Giving up "
                               "in %.0f s.", self.what, hint,
                               BUILD_LOCK_TIMEOUT_S - (now - start))
                next_report = now + _LOCK_REPORT_S
            time.sleep(_LOCK_POLL_S)
        try:
            with os.fdopen(fd, "w") as f:
                f.write(f"pid {os.getpid()} on {socket.gethostname()}")
            yield
        finally:
            with contextlib.suppress(FileNotFoundError):
                os.remove(path)


class LazyExtension:
    """Loads one extension on first use; a build error is kept and re-raised, an interrupt is not."""

    def __init__(self, extension: Extension, op_names: Tuple[str, ...],
                 namespace: Optional[str] = None,
                 on_load: Optional[Callable[[], None]] = None):
        self.extension = extension
        self.op_names = op_names
        self.namespace = namespace          # torch.ops namespace; None for a pybind11 module
        self._on_load = on_load
        self._lock = threading.Lock()
        self._handle: Optional[Any] = None
        self._load_error: Optional[Exception] = None
        self.library_path: Optional[str] = None

    def is_loaded(self) -> bool:
        return self._handle is not None

    def load(self, verbose: bool = False) -> Any:
        """Compile (if needed) and load; returns the ops namespace or the module."""
        if self._handle is not None:
            return self._handle
        ext = self.extension
        with self._lock:
            if self._handle is not None:
                return self._handle
            if self._load_error is not None:
                raise ext.error(
                    f"{ext.what} previously failed to load: {self._load_error}"
                ) from self._load_error
            try:
                handle = self._load(verbose)
            except ext.error:
                raise
            except Exception as e:  # noqa: BLE001 - surface build errors verbatim
                self._load_error = e
                raise ext.error(
                    f"failed to build/load the {ext.what} (no Triton fallback). "
                    f"Original error: {type(e).__name__}: {e}"
                ) from e
            self._handle = handle
            return handle

    def _load(self, verbose: bool) -> Any:
        module, self.library_path = self.extension.build_and_load(verbose=verbose)
        handle = module if self.namespace is None else getattr(torch.ops, self.namespace)
        missing = [n for n in self.op_names if not hasattr(handle, n)]
        if missing:
            where = "it" if self.namespace is None else f"torch.ops.{self.namespace}"
            raise self.extension.error(f"{self.extension.what} loaded but {where} lacks {missing}")
        if self._on_load is not None:
            self._on_load()
        return handle


def _import_module(name: str, library: str) -> Any:
    """Import the extension module ``name`` from the shared library at ``library``."""
    spec = importlib.util.spec_from_file_location(name, library)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@contextlib.contextmanager
def _environ(name: str, value: str) -> Iterator[None]:
    """Set one environment variable for the duration of the block."""
    previous = os.environ.get(name)
    os.environ[name] = value
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous


def _read_text(path: str) -> str:
    try:
        return Path(path).read_text().strip()
    except OSError:
        return ""
