"""Ahead-of-time build of the CUDA extensions: python -m omnimem.nsa.cuda_backend.build [--check]"""
from __future__ import annotations

import argparse
import sys
import time

import torch


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Pre-compile the OmniMem CUDA extensions")
    parser.add_argument("--quiet", action="store_true", help="hide compiler output")
    parser.add_argument("--check", action="store_true", help="only check toolchain / print keys")
    parser.add_argument("--nsa", action="store_true", help="build only the NSA extension")
    parser.add_argument("--block-kv", action="store_true",
                        help="build only the block-KV cache extension")
    args = parser.parse_args(argv)

    from omnimem.nsa.cuda_backend import _block_kv_loader, _loader

    loaders = []
    if args.nsa or not args.block_kv:
        loaders.append(_loader)
    if args.block_kv or not args.nsa:
        loaders.append(_block_kv_loader)
    for loader in loaders:
        extension = loader.EXTENSION
        cuda_home = extension.check_toolchain()
        arch = extension.arch_list()
        print(f"torch={torch.__version__} cuda={torch.version.cuda} CUDA_HOME={cuda_home}")
        print(f"arch_list={arch} extension={extension.name(arch)}")
        if args.check:
            continue
        t0 = time.time()
        loader.load(verbose=not args.quiet)
        print(f"loaded {loader.library_path()} in {time.time() - t0:.1f}s")
        print("registered ops:", ", ".join(loader.OP_NAMES))
    return 0


if __name__ == "__main__":
    sys.exit(main())
