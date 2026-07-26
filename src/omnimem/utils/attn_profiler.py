"""Triple-attention profiler for CausalNSSelfAttention using CUDA events."""
from __future__ import annotations
import importlib
from collections import defaultdict

import torch
from einops import rearrange


class TripleAttnProfiler:
    def __init__(self):
        self.events = defaultdict(list)  # (layer_idx, kind) -> [(start, end), ...]
        self.shapes = defaultdict(list)  # (layer_idx, kind) -> [k_seq_len, ...]
        self.fwd_events = []  # [(start_event, end_event), ...]
        self._original_forward = None
        self._target_cls = None
        self._original_slc = None
        self._original_model_forward = None
        self._target_model_cls = None

    def attach(self, model_module: str = "omnimem.models.causal_wan_nsa_model"):
        M = importlib.import_module(model_module)
        target_cls = M.CausalNSSelfAttention
        attention_fn = M.attention
        rope_fn = M.causal_reorder_rope_apply

        self._target_cls = target_cls
        self._original_forward = target_cls.forward
        prof = self

        def patched(self, x, seq_lens, grid_sizes, freqs,
                    start_frame=0, kv_cache=None, layer_idx=0, temb=None):
            b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)

            if temb is not None and self.use_temb:
                t_ = temb.reshape(b, -1, temb.shape[-1])
                if t_.shape[1] == 1:
                    t_ = t_.expand(b, s, -1)
                x_t = torch.cat([x, t_], dim=-1)
            else:
                x_t = x
            # gate
            g = self.g_proj(x_t)
            g = rearrange(g, '... (h d) -> ... h d', d=3)
            g = torch.clamp(g, min=-10.0, max=10.0)
            if self.gate_mode == 'sigmoid':
                g_cmp, g_slc, g_swa = g.sigmoid().unbind(3)
            else:
                g_cmp, g_slc, g_swa = g.softmax(dim=-1).unbind(3)

            # rope
            rk = dict(pool_f=self.pool_f, pool_h=self.pool_h, pool_w=self.pool_w)
            roped_q = rope_fn(q, grid_sizes, freqs, start_frame=start_frame, **rk).type_as(v)
            roped_k = rope_fn(k, grid_sizes, freqs, start_frame=start_frame, **rk).type_as(v)

            h_t = grid_sizes[0, 1].to(torch.long)
            w_t = grid_sizes[0, 2].to(torch.long)
            sf = torch.as_tensor(start_frame, device=grid_sizes.device, dtype=torch.long)
            start_id = h_t * w_t * sf

            # cache updates — timed as one bucket 'cache'
            ds = torch.cuda.default_stream()
            s_ev = torch.cuda.Event(enable_timing=True)
            e_ev = torch.cuda.Event(enable_timing=True)
            s_ev.record(ds)
            roped_k_cmp = kv_cache.update_cache(
                name='k_cmp_cache',
                hidden_state=roped_k.view(b, s // self.kv_block_size,
                                          self.kv_block_size, n, d).mean(2),
                layer_idx=layer_idx, start_id=start_id // self.kv_block_size)
            v_cmp = kv_cache.update_cache(
                name='v_cmp_cache',
                hidden_state=v.view(b, s // self.kv_block_size,
                                    self.kv_block_size, n, d).mean(2),
                layer_idx=layer_idx, start_id=start_id // self.kv_block_size)
            roped_k_swa = kv_cache.update_cache(
                name='k_cache', hidden_state=roped_k,
                layer_idx=layer_idx, start_id=start_id)
            v_swa = kv_cache.update_cache(
                name='v_cache', hidden_state=v,
                layer_idx=layer_idx, start_id=start_id)
            e_ev.record(ds)
            prof.events[(self.layer_idx, 'cache')].append((s_ev, e_ev))
            prof.shapes[(self.layer_idx, 'cache')].append(s)

            def _time(kind, k_len, fn):
                ds = torch.cuda.default_stream()
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                if kind == 'slc':
                    torch.cuda.synchronize()
                s_ev.record(ds)
                out = fn()
                if kind == 'slc':
                    torch.cuda.synchronize()
                e_ev.record(ds)
                prof.events[(self.layer_idx, kind)].append((s_ev, e_ev))
                prof.shapes[(self.layer_idx, kind)].append(k_len)
                return out

            o_cmp = _time('cmp', roped_k_cmp.shape[1],
                          lambda: attention_fn(q=roped_q, k=roped_k_cmp, v=v_cmp))
            o_swa = _time('swa', roped_k_swa.shape[1],
                          lambda: attention_fn(q=roped_q, k=roped_k_swa, v=v_swa))

            o_slc = None
            if self.skip_module != 'slc':
                q_chunk_offset = int(start_id) // s
                o_slc = _time('slc', roped_k_cmp.shape[1],
                              lambda: self.slc_attention_padded_ptr(
                                  roped_query=roped_q,
                                  roped_key_cmp=roped_k_cmp,
                                  kv_cache=kv_cache,
                                  q_chunk_offset=q_chunk_offset))

            if self.skip_module == 'cmp':
                x = g_slc.unsqueeze(-1) * o_slc + g_swa.unsqueeze(-1) * o_swa
            elif self.skip_module == 'slc':
                x = g_cmp.unsqueeze(-1) * o_cmp + g_swa.unsqueeze(-1) * o_swa
            else:
                x = (g_cmp.unsqueeze(-1) * o_cmp
                     + g_slc.unsqueeze(-1) * o_slc
                     + g_swa.unsqueeze(-1) * o_swa)

            x = x.flatten(2)
            x = self.o(x)
            return x

        target_cls.forward = patched

        # patch slc_attention_padded_ptr to break into topk + attn
        self._original_slc = target_cls.slc_attention_padded_ptr
        should_use_fn = M.should_use_selection_attention
        topk_fn = M.parallel_nsa_topk_grouped_heads
        sel_attn_fn = M.selection_attention_padded_ptr_fast

        def patched_slc(self, roped_query, roped_key_cmp, kv_cache, q_chunk_offset=0):
            s_arg = roped_query.shape[1]

            if not (self.progressive_exclude or should_use_fn(
                seq_len=roped_key_cmp.shape[1] * self.kv_block_size,
                chunk_size=s_arg,
                exclude_window_chunks=self.exclude_window_chunks,
                exclude_sink_chunks=self.exclude_sink_chunks,
            )):
                return torch.zeros_like(roped_query)

            ds = torch.cuda.default_stream()

            # time topk
            s_topk = torch.cuda.Event(enable_timing=True)
            e_topk = torch.cuda.Event(enable_timing=True)
            s_topk.record(ds)
            block_indices = topk_fn(
                q=roped_query, k=roped_key_cmp,
                num_kv_head_groups=self.num_kv_head_groups,
                block_counts=self.num_selected_blocks,
                block_size=self.kv_block_size,
                chunk_size=s_arg,
                causal=False,
                group_size=self.group_q_size,
                exclude_window_chunks=self.exclude_window_chunks,
                exclude_sink_chunks=self.exclude_sink_chunks,
                progressive_exclude=self.progressive_exclude,
                q_chunk_offset=q_chunk_offset,
            ).to(torch.int32).contiguous()
            e_topk.record(ds)
            prof.events[(self.layer_idx, 'slc_topk')].append((s_topk, e_topk))

            meta_k = kv_cache.get_chunk_metadata(self.layer_idx, 'k_cache')
            meta_v = kv_cache.get_chunk_metadata(self.layer_idx, 'v_cache')
            if meta_k is None or meta_v is None:
                return torch.zeros_like(roped_query)

            ptrs_k, n_chunks_k, strides = meta_k
            ptrs_v, _, _ = meta_v

            # time selection attn
            s_attn = torch.cuda.Event(enable_timing=True)
            e_attn = torch.cuda.Event(enable_timing=True)
            s_attn.record(ds)
            out = sel_attn_fn(
                q=roped_query,
                block_indices=block_indices,
                block_size=self.kv_block_size,
                group_size=self.group_q_size,
                chunk_len=strides['chunk_len'],
                chunk_base_ptrs_k=ptrs_k,
                chunk_base_ptrs_v=ptrs_v,
                n_chunks_per_head=n_chunks_k,
                chunk_strides=strides,
                verify_complete=kv_cache._has_cpu_chunks,
                kv_cache=kv_cache,
                layer_idx=self.layer_idx,
            )
            e_attn.record(ds)
            prof.events[(self.layer_idx, 'slc_attn')].append((s_attn, e_attn))

            return out

        target_cls.slc_attention_padded_ptr = patched_slc

        # patch model-level forward
        model_cls = (getattr(M, 'CausalWanNSAModel', None)
                     or getattr(M, 'CausalWanModel', None))
        if model_cls is not None:
            self._target_model_cls = model_cls
            self._original_model_forward = model_cls.forward
            orig_fwd = model_cls.forward

            def patched_model(self, *args, **kwargs):
                ds = torch.cuda.default_stream()
                s_ev = torch.cuda.Event(enable_timing=True)
                e_ev = torch.cuda.Event(enable_timing=True)
                s_ev.record(ds)
                out = orig_fwd(self, *args, **kwargs)
                e_ev.record(ds)
                prof.fwd_events.append((s_ev, e_ev))
                return out

            model_cls.forward = patched_model

    def detach(self):
        if self._target_cls is not None:
            self._target_cls.forward = self._original_forward
            if self._original_slc is not None:
                self._target_cls.slc_attention_padded_ptr = self._original_slc
            self._target_cls = None
            self._original_forward = None
            self._original_slc = None
        if self._target_model_cls is not None:
            self._target_model_cls.forward = self._original_model_forward
            self._target_model_cls = None
            self._original_model_forward = None

    def reset(self):
        self.events.clear()
        self.shapes.clear()
        self.fwd_events.clear()

    def _aggregate(self):
        """Sync CUDA events and aggregate timings by kind and layer."""
        torch.cuda.synchronize()
        by_kind = defaultdict(list)
        by_layer = defaultdict(lambda: defaultdict(list))
        skipped_by_kind = defaultdict(int)
        err_sample = {}
        for (li, kind), pairs in self.events.items():
            for s_ev, e_ev in pairs:
                try:
                    ms = s_ev.elapsed_time(e_ev)
                except (ValueError, RuntimeError) as ex:
                    skipped_by_kind[kind] += 1
                    err_sample.setdefault(kind, str(ex))
                    continue
                by_kind[kind].append(ms)
                by_layer[li][kind].append(ms)
        if skipped_by_kind:
            for k, n in skipped_by_kind.items():
                print(f"[profiler] {k}: skipped {n} pairs — first error: {err_sample.get(k, '?')}")
        return by_kind, by_layer

    def report(self, show_layers: bool = True, show_shapes: bool = True):
        by_kind, by_layer = self._aggregate()
        if not by_kind and not self.fwd_events:
            print("[profiler] no events recorded")
            return

        fwd_ms = []
        for s_ev, e_ev in self.fwd_events:
            try:
                fwd_ms.append(s_ev.elapsed_time(e_ev))
            except (ValueError, RuntimeError):
                pass
        fwd_total = sum(fwd_ms)
        if fwd_ms:
            fs = sorted(fwd_ms)
            print("=" * 80)
            print(f"Forward summary   #forwards={len(fs)}")
            print("=" * 80)
            print(f"{'mean':>9} {'p50':>9} {'p95':>9} {'min':>9} {'max':>9} {'total':>11}")
            print("-" * 80)
            print(f"{fwd_total/len(fs):>9.2f} {fs[len(fs)//2]:>9.2f} "
                  f"{fs[min(int(len(fs)*0.95), len(fs)-1)]:>9.2f} "
                  f"{fs[0]:>9.2f} {fs[-1]:>9.2f} {fwd_total:>11.1f}")
            print()

        print("=" * 80)
        print(f"Triple-attention summary   layers={len(by_layer)}   "
              f"forwards/layer≈{len(next(iter(by_layer.values())).get('cmp', []))}")
        print("=" * 80)
        print(f"{'kind':<6} {'#calls':>7} {'mean':>9} {'p50':>9} {'p95':>9} "
              f"{'min':>9} {'max':>9} {'total':>11}")
        print("-" * 80)
        total = 0.0
        for kind in ('cmp', 'swa', 'slc', 'cache'):
            vs = sorted(by_kind.get(kind, []))
            if not vs:
                continue
            tot = sum(vs)
            total += tot
            mean = tot / len(vs)
            p50 = vs[len(vs) // 2]
            p95 = vs[min(int(len(vs) * 0.95), len(vs) - 1)]
            print(f"{kind:<9} {len(vs):>7} {mean:>9.3f} {p50:>9.3f} {p95:>9.3f} "
                  f"{vs[0]:>9.3f} {vs[-1]:>9.3f} {tot:>11.1f}")
        for kind in ('slc_topk', 'slc_attn'):
            vs = sorted(by_kind.get(kind, []))
            if not vs:
                continue
            tot = sum(vs)
            mean = tot / len(vs)
            p50 = vs[len(vs) // 2]
            p95 = vs[min(int(len(vs) * 0.95), len(vs) - 1)]
            print(f"  {kind:<7} {len(vs):>7} {mean:>9.3f} {p50:>9.3f} {p95:>9.3f} "
                  f"{vs[0]:>9.3f} {vs[-1]:>9.3f} {tot:>11.1f}")
        print("-" * 80)
        print(f"{'SUM':<6} {'':>7} {'':>9} {'':>9} {'':>9} {'':>9} {'':>9} {total:>11.1f} ms")
        print()
        if fwd_total > 0:
            print("Share of total forward time:")
            for kind in ('cmp', 'swa', 'slc', 'cache'):
                if kind in by_kind:
                    tot = sum(by_kind[kind])
                    print(f"  {kind:<6}: {tot:>9.1f} ms   ({100 * tot / fwd_total:>5.1f}%)")
            other = fwd_total - total
            print(f"  {'other':<6}: {other:>9.1f} ms   ({100 * other / fwd_total:>5.1f}%)  "
                  f"<- qkv proj / ffn / cross-attn / norm / gate / etc.")

        else:
            print("Share of profiled time:")
            for kind in ('cmp', 'swa', 'slc', 'cache'):
                if kind in by_kind:
                    tot = sum(by_kind[kind])
                    print(f"  {kind:<6}: {tot:>9.1f} ms   ({100 * tot / total:>5.1f}%)")
        print()

        if show_shapes:
            print("KV seq-len observed (k.shape[1]) — first sample per kind:")
            seen = set()
            for (li, kind), lens in self.shapes.items():
                if kind in seen or not lens:
                    continue
                seen.add(kind)
                print(f"  {kind}: {lens[0]}  (range {min(lens)} .. {max(lens)})")
            print()

        if show_layers:
            layers = sorted(by_layer.keys())
            print(f"Per-layer mean (ms)")
            print("-" * 68)
            print(f"{'layer':>6} {'cmp':>10} {'swa':>10} {'slc':>10} {'cache':>10} {'sum':>10}")

            def avg(d, k):
                vs = d.get(k, [])
                return sum(vs) / len(vs) if vs else 0.0

            agg_cm = agg_sw = agg_sl = agg_ca = 0.0
            for li in layers:
                r = by_layer[li]
                cm, sw, sl, ca = avg(r, 'cmp'), avg(r, 'swa'), avg(r, 'slc'), avg(r, 'cache')
                agg_cm += cm; agg_sw += sw; agg_sl += sl; agg_ca += ca
                print(f"{li:>6} {cm:>10.3f} {sw:>10.3f} {sl:>10.3f} {ca:>10.3f} "
                      f"{cm+sw+sl+ca:>10.3f}")
            print("-" * 68)
            n = len(layers)
            print(f"{'avg':>6} {agg_cm/n:>10.3f} {agg_sw/n:>10.3f} {agg_sl/n:>10.3f} "
                  f"{agg_ca/n:>10.3f} {(agg_cm+agg_sw+agg_sl+agg_ca)/n:>10.3f}")
            print(f"{'×L':>6} {agg_cm:>10.3f} {agg_sw:>10.3f} {agg_sl:>10.3f} "
                  f"{agg_ca:>10.3f} {agg_cm+agg_sw+agg_sl+agg_ca:>10.3f}  ms / forward")
        print()