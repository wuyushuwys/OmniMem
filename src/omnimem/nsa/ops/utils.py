# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Argsort via bitonic sort. Adapted from https://github.com/triton-lang/triton/issues/3698#issuecomment-2067681396


import triton
import triton.language as tl


@triton.jit
def _compare_and_swap(
        x,
        ids,
        flip,
        i: tl.constexpr,
        n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    shape: tl.constexpr = [n_outer * 2 ** i, 2, 2 ** (n_dims - i - 1)]
    y = tl.reshape(x, shape)
    # slice left/right with 'stride' 2**(n_dims - i - 1)
    mask = tl.arange(0, 2)[None, :, None]
    left = tl.broadcast_to(tl.sum(y * (1 - mask), 1)[:, None, :], shape).to(y.dtype)
    right = tl.broadcast_to(tl.sum(y * mask, 1)[:, None, :], shape).to(y.dtype)
    left = tl.reshape(left, x.shape)
    right = tl.reshape(right, x.shape)
    # idx
    y_idx = tl.reshape(ids, shape)
    left_idx = tl.broadcast_to(tl.sum(y_idx * (1 - mask), 1)[:, None, :], shape)
    right_idx = tl.broadcast_to(tl.sum(y_idx * mask, 1)[:, None, :], shape)
    left_idx = tl.reshape(left_idx, x.shape).to(y_idx.dtype)
    right_idx = tl.reshape(right_idx, x.shape).to(y_idx.dtype)
    # actual compare-and-swap
    idtype = tl.core.get_int_dtype(bitwidth=x.dtype.primitive_bitwidth, signed=True)
    ileft = left.to(idtype, bitcast=True)
    iright = right.to(idtype, bitcast=True)
    ix = x.to(idtype, bitcast=True)

    cond = (left > right) != flip
    ret = ix ^ tl.where(cond, ileft ^ iright, tl.zeros_like(ix))
    new_ids = ids ^ tl.where(cond, left_idx ^ right_idx, tl.zeros_like(ids))
    return ret.to(x.dtype, bitcast=True), new_ids


@triton.jit
def _bitonic_merge(
        x,
        ids,
        stage: tl.constexpr,
        order: tl.constexpr,
        n_dims: tl.constexpr,
):
    n_outer: tl.constexpr = x.numel >> n_dims
    tl.static_assert(stage <= n_dims)
    # flip: 0 = sort ascending, 1 = alternating (every 2 elements flip direction)
    if order == 2:
        shape: tl.constexpr = [n_outer * 2 ** (n_dims - 1 - stage), 2, 2 ** stage]
        flip = tl.reshape(tl.broadcast_to(tl.arange(0, 2)[None, :, None], shape), x.shape)
    else:
        flip = order
    for i in tl.static_range(stage):
        x, ids = _compare_and_swap(x, ids, flip, i + (n_dims - stage), n_dims)
    return x, ids


@triton.jit
def argsort(
        x,
        ids,
        dim: tl.constexpr = None,
        descending: tl.constexpr = tl.core.CONSTEXPR_0,
):
    _dim: tl.constexpr = len(x.shape) - 1 if dim is None else dim
    tl.static_assert(_dim == len(x.shape) - 1, "only minor dimension is currently supported")

    n_dims: tl.constexpr = tl.standard._log2(x.shape[_dim])

    for i in tl.static_range(1, n_dims + 1):
        x, ids = _bitonic_merge(x, ids, i, 2 if i < n_dims else descending, n_dims)
    return x, ids
