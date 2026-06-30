"""NUMA-aware CPU affinity binding for GPU workloads on multi-socket servers."""

import os
import re
import subprocess
import sys
import warnings
from typing import Optional, Set, Dict, Any


def _read_node_cpulist(node: int) -> Optional[Set[int]]:
    """Parse /sys/.../nodeN/cpulist (e.g. '0-23,48-71') into a set of CPU ids."""
    path = f"/sys/devices/system/node/node{node}/cpulist"
    try:
        with open(path) as f:
            spec = f.read().strip()
    except OSError:
        return None
    if not spec:
        return None

    cpus: Set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            cpus.update(range(int(lo), int(hi) + 1))
        else:
            cpus.add(int(part))
    return cpus or None


def _detect_gpu_numa(gpu_id: int) -> Optional[int]:
    """Return the NUMA node for given GPU via nvidia-smi, or None if unavailable."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "topo", "-m"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=5,
        )
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return None

    for line in out.splitlines():
        if not re.match(rf"^GPU{gpu_id}\b", line):
            continue
        parts = line.split()
        for tok in reversed(parts[1:]):
            if tok == "N/A":
                continue
            if tok.isdigit():
                v = int(tok)
                if 0 <= v < 16:
                    return v
        return None
    return None


def _resolve_gpu_id(explicit: Optional[int]) -> Optional[int]:
    """Resolve GPU id from explicit arg, LOCAL_RANK, CUDA_VISIBLE_DEVICES, or default 0."""
    if explicit is not None:
        return explicit

    local_rank = os.environ.get("LOCAL_RANK")
    if local_rank is not None and local_rank.isdigit():
        return int(local_rank)

    cuda_dev = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cuda_dev is not None:
        cuda_dev = cuda_dev.strip()
        if cuda_dev in ("", "-1"):
            return None
        first = cuda_dev.split(",")[0].strip()
        if first.isdigit():
            return int(first)
        return None

    return 0


def _count_numa_nodes() -> int:
    """Count NUMA nodes on this machine. Returns 1 on failure or single-NUMA."""
    try:
        nodes = [
            d for d in os.listdir("/sys/devices/system/node")
            if d.startswith("node") and d[4:].isdigit()
        ]
        return max(len(nodes), 1)
    except OSError:
        return 1


class NumaBindResult:
    """Outcome of a bind_to_gpu_numa() call. Truthy if binding succeeded."""

    __slots__ = ("ok", "gpu_id", "node", "n_cpus", "reason")

    def __init__(
        self,
        ok: bool,
        gpu_id: Optional[int] = None,
        node: Optional[int] = None,
        n_cpus: int = 0,
        reason: str = "",
    ):
        self.ok = ok
        self.gpu_id = gpu_id
        self.node = node
        self.n_cpus = n_cpus
        self.reason = reason

    def __bool__(self) -> bool:
        return self.ok

    def __repr__(self) -> str:
        if self.ok:
            return (f"NumaBindResult(ok=True, gpu={self.gpu_id}, "
                    f"node={self.node}, cpus={self.n_cpus})")
        return f"NumaBindResult(ok=False, reason={self.reason!r})"


def bind_to_gpu_numa(
    gpu_id: Optional[int] = None,
    verbose: bool = True,
    warn_on_failure: bool = True,
) -> NumaBindResult:
    """Pin process CPU affinity to the NUMA node of the given GPU.

    Args:
        gpu_id: GPU to align to; auto-detected if None.
        verbose: Print status to stderr.
        warn_on_failure: Warn if binding fails on a multi-NUMA machine.

    Returns:
        NumaBindResult (truthy if binding succeeded).
    """
    if os.environ.get("DISABLE_NUMA_BIND"):
        if verbose:
            print("[numa] DISABLE_NUMA_BIND set, skipping", file=sys.stderr)
        return NumaBindResult(ok=False, reason="disabled by env")

    if not hasattr(os, "sched_setaffinity"):
        return NumaBindResult(
            ok=False, reason="sched_setaffinity unavailable (non-Linux)",
        )

    n_nodes = _count_numa_nodes()
    if n_nodes <= 1:
        if verbose:
            print("[numa] single-NUMA machine, no binding needed",
                  file=sys.stderr)
        return NumaBindResult(
            ok=False, reason="single-NUMA machine (no perf impact)",
        )

    target_gpu = _resolve_gpu_id(gpu_id)
    if target_gpu is None:
        return NumaBindResult(ok=False, reason="no GPU selected")

    node = _detect_gpu_numa(target_gpu)
    if node is None:
        msg = (
            f"could not detect NUMA node for GPU {target_gpu} "
            f"(machine has {n_nodes} NUMA nodes; nvidia-smi may be "
            f"unavailable or the GPU has no reported affinity). "
            f"PCIe transfers may go cross-socket, costing 10-30% on "
            f"throughput-bound workloads."
        )
        if warn_on_failure:
            warnings.warn(f"[numa] {msg}", RuntimeWarning, stacklevel=2)
        elif verbose:
            print(f"[numa] {msg}", file=sys.stderr)
        return NumaBindResult(
            ok=False, gpu_id=target_gpu, reason="GPU NUMA detection failed",
        )

    cpus = _read_node_cpulist(node)
    if not cpus:
        msg = (
            f"detected GPU {target_gpu} on NUMA Node {node}, but could "
            f"not read CPU list from /sys (likely running in a container "
            f"without /sys access). Mount /sys read-only or set "
            f"DISABLE_NUMA_BIND=1 to silence. PCIe transfers may go "
            f"cross-socket."
        )
        if warn_on_failure:
            warnings.warn(f"[numa] {msg}", RuntimeWarning, stacklevel=2)
        elif verbose:
            print(f"[numa] {msg}", file=sys.stderr)
        return NumaBindResult(
            ok=False, gpu_id=target_gpu, node=node,
            reason="CPU list unreadable",
        )

    try:
        os.sched_setaffinity(0, cpus)
    except OSError as e:
        msg = (
            f"sched_setaffinity({len(cpus)} CPUs) failed: {e}. "
            f"Process likely lacks the capability to set affinity "
            f"(restricted cgroup, missing CAP_SYS_NICE, etc.). "
            f"PCIe transfers may go cross-socket."
        )
        if warn_on_failure:
            warnings.warn(f"[numa] {msg}", RuntimeWarning, stacklevel=2)
        elif verbose:
            print(f"[numa] {msg}", file=sys.stderr)
        return NumaBindResult(
            ok=False, gpu_id=target_gpu, node=node,
            reason=f"sched_setaffinity failed: {e}",
        )

    if verbose:
        print(
            f"[numa] GPU {target_gpu} on Node {node}, "
            f"bound to {len(cpus)} CPUs",
            file=sys.stderr,
        )
    return NumaBindResult(
        ok=True, gpu_id=target_gpu, node=node, n_cpus=len(cpus),
    )


def numa_status() -> Dict[str, Any]:
    """Return NUMA/affinity status dict for current process."""
    out: Dict[str, Any] = {
        "n_nodes": _count_numa_nodes(),
        "gpu_id": None,
        "gpu_numa_node": None,
        "bound_cpus": 0,
        "bound_to_node": None,
        "cross_numa": False,
    }

    target_gpu = _resolve_gpu_id(None)
    out["gpu_id"] = target_gpu
    if target_gpu is not None:
        out["gpu_numa_node"] = _detect_gpu_numa(target_gpu)

    if hasattr(os, "sched_getaffinity"):
        try:
            mask = os.sched_getaffinity(0)
            out["bound_cpus"] = len(mask)

            node_hits: Set[int] = set()
            for n in range(out["n_nodes"]):
                cpus = _read_node_cpulist(n)
                if cpus and mask & cpus:
                    node_hits.add(n)
            if len(node_hits) == 1:
                out["bound_to_node"] = next(iter(node_hits))
        except OSError:
            pass

    if (out["gpu_numa_node"] is not None
            and out["bound_to_node"] is not None
            and out["gpu_numa_node"] != out["bound_to_node"]):
        out["cross_numa"] = True

    return out


def print_numa_status(file=sys.stderr) -> None:
    """One-line human-readable NUMA status."""
    s = numa_status()
    if s["n_nodes"] <= 1:
        print("[numa] single-NUMA machine", file=file)
        return
    gpu = s["gpu_id"]
    gpu_node = s["gpu_numa_node"]
    bound = s["bound_to_node"]
    n_cpus = s["bound_cpus"]
    if s["cross_numa"]:
        print(
            f"[numa] CROSS-NUMA: GPU {gpu} on Node {gpu_node}, "
            f"process on Node {bound} ({n_cpus} CPUs). "
            f"Expect slow PCIe.",
            file=file,
        )
    elif bound is not None and gpu_node is not None:
        print(
            f"[numa] aligned: GPU {gpu} ↔ Node {bound} ({n_cpus} CPUs)",
            file=file,
        )
    else:
        print(
            f"[numa] partial info: gpu={gpu} gpu_node={gpu_node} "
            f"bound_node={bound} bound_cpus={n_cpus}",
            file=file,
        )


if __name__ == "__main__":
    print(f"NUMA nodes detected: {_count_numa_nodes()}")
    print()
    print("Per-GPU NUMA mapping:")
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,name", "--format=csv,noheader"],
            text=True,
        )
    except Exception as e:
        print(f"  nvidia-smi unavailable: {e}")
        sys.exit(1)

    for line in out.strip().splitlines():
        idx_s, name = line.split(",", 1)
        idx = int(idx_s)
        node = _detect_gpu_numa(idx)
        cpus = _read_node_cpulist(node) if node is not None else None
        n_cpus = len(cpus) if cpus else 0
        print(f"  GPU {idx} ({name.strip()}): Node {node}, {n_cpus} CPUs")

    print()
    print("Current process status:")
    print_numa_status(file=sys.stdout)