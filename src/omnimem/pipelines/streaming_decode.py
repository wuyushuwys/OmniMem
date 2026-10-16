"""Streaming latent -> RGB delivery for single-video low-latency inference.

The sampler submits final latent chunks; a decoder thread decodes them and a writer thread feeds a FrameSink.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Protocol, Tuple

import torch

from omnimem.utils.stage_marks import StageMarks
from omnimem.utils.torch_utils import indexed_device
from omnimem.utils.validation_prompts import decoded_frames

__all__ = [
    "LatentChunk",
    "LatentGenerationOutput",
    "DecodedChunk",
    "SessionError",
    "FrameSink",
    "WanChunkDecoder",
    "rgb_uint8_from_decoded",
    "output_frames_for_latents",
    "decode_latents_chunked_to_cpu",
    "decoder_thread_streams",
    "StreamingSession",
    "SessionResult",
]


@dataclass
class LatentChunk:
    """One finished latent chunk: an owned snapshot plus a ``ready_event`` to wait on."""

    chunk_index: int
    latent: torch.Tensor
    ready_event: Optional[torch.cuda.Event]
    submit_time: float


@dataclass
class LatentGenerationOutput:
    """Result of ``generate(..., return_latents=True)`` (no VAE decode performed)."""

    num_chunks: int


@dataclass
class DecodedChunk:
    chunk_index: int
    frames: torch.Tensor  # CPU uint8 [T, H, W, 3]
    release: Optional[Callable[[], None]] = None


class SessionError(RuntimeError):
    """Raised on the caller's thread when any stage of the session failed."""


class FrameSink(Protocol):
    def on_frames(self, frames_u8: torch.Tensor, frame_start: int, chunk_index: int) -> None: ...
    def finish(self) -> Any: ...
    def abort(self) -> None: ...


def output_frames_for_latents(num_latents: int, first_chunk: bool) -> int:
    """Wan VAE temporal rule: first latent -> 1 frame, every later latent -> 4 frames."""
    if num_latents <= 0:
        return 0
    return decoded_frames(num_latents) if first_chunk else 4 * num_latents


def rgb_uint8_from_decoded(video: torch.Tensor) -> torch.Tensor:
    """``[1, 3, T, H, W]`` decoder output -> ``[T, H, W, 3]`` uint8, same ops as ``tensor2vid``."""
    assert video.dim() == 5 and video.shape[0] == 1, f"expected [1,3,T,H,W], got {tuple(video.shape)}"
    vid = video[0].float().permute(1, 0, 2, 3).clamp(-1, 1)
    vid = vid * 127.5 + 127.5
    return vid.to(torch.uint8).permute(0, 2, 3, 1).contiguous()


class WanChunkDecoder:
    """Request-scoped stateful decode of consecutive latent chunks with the Wan VAE."""

    def __init__(self, vae, device: torch.device, autocast_dtype: torch.dtype = torch.bfloat16,
                 fast: Optional[bool] = False):
        self.vae = vae
        self.device = indexed_device(device)
        self.autocast_dtype = autocast_dtype
        self.fast = None if fast is None else bool(fast)
        self._latents_mean: Optional[torch.Tensor] = None
        self._latents_std: Optional[torch.Tensor] = None
        self.started = False

    def start(self) -> None:
        vae = self.vae
        assert vae.device == self.device, f"VAE is on {vae.device}, session expects {self.device}"
        if self.fast is not None:
            if hasattr(vae, "enable_fast_decode"):
                vae.enable_fast_decode(self.fast)
                if self.device.type == "cuda":
                    # Re-laid weights must be final before start returns.
                    torch.cuda.current_stream(self.device).synchronize()
            elif self.fast:
                raise TypeError(f"{type(vae).__name__} has no enable_fast_decode(); "
                                f"fast=True unsupported")
        vae.clear_cache()
        self._latents_mean = torch.tensor(
            vae.config.latents_mean, device=vae.device, dtype=vae.dtype).reshape(1, -1, 1, 1, 1)
        self._latents_std = torch.tensor(
            vae.config.latents_std, device=vae.device, dtype=vae.dtype).reshape(1, -1, 1, 1, 1)
        self.started = True

    def decode(self, z_normalized: torch.Tensor) -> torch.Tensor:
        """``[1, C, f, h, w]`` normalised latents -> ``[1, 3, T, H, W]`` in [-1, 1]."""
        assert self.started, "call start() first"
        assert z_normalized.device == self.device, f"latent on {z_normalized.device}, expected {self.device}"
        z = z_normalized * self._latents_std + self._latents_mean
        with torch.autocast("cuda", dtype=self.autocast_dtype):
            out = self.vae.streaming_decode(z).sample
        return out

    def finish(self) -> None:
        if self.started:
            self.vae.clear_cache()
        self.started = False


def decode_latents_chunked_to_cpu(vae, latents_normalized: torch.Tensor, chunk_size: int,
                                  autocast_dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Decode a whole latent tensor in temporal groups and return float CPU ``[1,3,T,H,W]``."""
    dec = WanChunkDecoder(vae, vae.device, autocast_dtype, fast=None)
    dec.start()
    outs = []
    try:
        T = latents_normalized.shape[2]
        for s in range(0, T, chunk_size):
            z = latents_normalized[:, :, s:s + chunk_size].to(vae.device)
            outs.append(dec.decode(z).float().cpu())
    finally:
        dec.finish()
    return torch.cat(outs, dim=2)


class _FailureState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.first_exc: Optional[BaseException] = None
        self.first_stage: Optional[str] = None
        self.cancel = threading.Event()

    def fail(self, stage: str, exc: BaseException) -> None:
        with self._lock:
            if self.first_exc is None:
                self.first_exc = exc
                self.first_stage = stage
        self.cancel.set()

    @property
    def failed(self) -> bool:
        return self.first_exc is not None

    def check(self) -> None:
        if self.first_exc is not None:
            raise SessionError(f"{self.first_stage} failed: {self.first_exc!r}") from self.first_exc
        if self.cancel.is_set():
            raise SessionError("session cancelled")


class _EOS:
    def __repr__(self) -> str:
        return "<EOS>"


EOS = _EOS()


class _BoundedQueue:
    """``queue.Queue`` with polling timeouts, cancel checks and blocking statistics."""

    def __init__(self, name: str, maxsize: int, failure: _FailureState, poll: float = 0.05):
        self.name = name
        self._q: "queue.Queue[Any]" = queue.Queue(maxsize=maxsize)
        self._failure = failure
        self._poll = poll
        self.peak = 0
        self.put_blocked_s = 0.0
        self.get_blocked_s = 0.0
        self.put_blocks = 0
        self._lock = threading.Lock()

    def put(self, item: Any, timeout: float) -> None:
        t0 = time.monotonic()
        blocked = False
        while True:
            self._failure.check()
            try:
                self._q.put(item, timeout=self._poll)
                break
            except queue.Full:
                blocked = True
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError(f"{self.name}: put timed out after {timeout:.0f}s (consumer stalled)")
        dt = time.monotonic() - t0
        with self._lock:
            self.peak = max(self.peak, self._q.qsize())
            if blocked:
                self.put_blocked_s += dt
                self.put_blocks += 1

    def get(self, timeout: float) -> Any:
        t0 = time.monotonic()
        while True:
            self._failure.check()
            try:
                item = self._q.get(timeout=self._poll)
                break
            except queue.Empty:
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError(f"{self.name}: get timed out after {timeout:.0f}s (producer stalled)")
        with self._lock:
            self.get_blocked_s += time.monotonic() - t0
        return item

    def qsize(self) -> int:
        return self._q.qsize()

    def stats(self) -> Dict[str, float]:
        return {"peak": self.peak, "put_blocked_s": self.put_blocked_s,
                "put_blocks": self.put_blocks, "get_blocked_s": self.get_blocked_s}


class _PinnedSlotPool:
    """Fixed pool of pinned host buffers for D2H; a slot is reusable only after ``release``."""

    def __init__(self, n_slots: int, shape, dtype, failure: _FailureState, poll: float = 0.05):
        self._free: "queue.Queue[torch.Tensor]" = queue.Queue()
        self.slots = [torch.empty(shape, dtype=dtype, pin_memory=True) for _ in range(n_slots)]
        for s in self.slots:
            self._free.put(s)
        self._failure = failure
        self._poll = poll
        self.acquire_blocked_s = 0.0
        self.bytes = sum(s.numel() * s.element_size() for s in self.slots)

    def acquire(self, timeout: float) -> torch.Tensor:
        t0 = time.monotonic()
        while True:
            self._failure.check()
            try:
                slot = self._free.get(timeout=self._poll)
                self.acquire_blocked_s += time.monotonic() - t0
                return slot
            except queue.Empty:
                if time.monotonic() - t0 > timeout:
                    raise TimeoutError("pinned D2H slot pool: acquire timed out (writer stalled)")

    def release(self, slot: torch.Tensor) -> None:
        self._free.put(slot)


def decoder_thread_streams(decode_device, dit_device) -> Tuple[torch.cuda.Stream, torch.cuda.Stream]:
    """A new ``(decode_stream, transfer_stream)`` pair for a session's decoder thread."""
    return torch.cuda.Stream(device=decode_device), torch.cuda.Stream(device=dit_device)


@dataclass
class SessionResult:
    frames_delivered: int
    sink_result: Any
    timeline: Dict[str, Any]
    chunk_records: List[Dict[str, float]]
    queue_stats: Dict[str, Dict[str, float]]
    d2h_pool_bytes: int


class StreamingSession:
    """Per-video decode/delivery session.  See module docstring."""

    def __init__(self, *, decoder, decode_device, dit_device, sink: FrameSink,
                 frame_hw, max_frames_per_chunk: int, queue_depth: int = 2,
                 decoder_streams: Optional[Tuple[torch.cuda.Stream, torch.cuda.Stream]] = None,
                 put_timeout: float = 900.0, get_timeout: float = 900.0,
                 join_timeout: float = 900.0, marks: Optional[StageMarks] = None,
                 autocast_dtype: torch.dtype = torch.bfloat16):
        self.decoder = decoder
        self.decode_device = indexed_device(decode_device)
        self.dit_device = indexed_device(dit_device)
        self.sink = sink
        self.frame_hw = tuple(frame_hw)
        self.max_frames_per_chunk = int(max_frames_per_chunk)
        self.queue_depth = int(queue_depth)
        if self.queue_depth < 1:
            raise ValueError(f"queue_depth must be >= 1, got {queue_depth}")
        self.decode_in_caller = self.decode_device == self.dit_device
        if decoder_streams is not None:
            on = (decoder_streams[0].device, decoder_streams[1].device)
            if on != (self.decode_device, self.dit_device):
                raise ValueError(
                    f"decoder_streams must be a (decode, transfer) pair on "
                    f"({self.decode_device}, {self.dit_device}), got streams on "
                    f"({on[0]}, {on[1]}): an event on another device's stream does not "
                    f"order this session's copies")
        self.decoder_streams = decoder_streams
        self.put_timeout = put_timeout
        self.get_timeout = get_timeout
        self.join_timeout = join_timeout
        self.marks = marks
        self.autocast_dtype = autocast_dtype

        self._failure = _FailureState()
        self._latent_q = _BoundedQueue("latent_queue", self.queue_depth, self._failure)
        self._frame_q = _BoundedQueue("frame_queue", self.queue_depth, self._failure)
        n_slots = self.queue_depth + 2
        self._pool = _PinnedSlotPool(
            n_slots, (self.max_frames_per_chunk, *self.frame_hw, 3), torch.uint8, self._failure)
        self._decoder_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._decoder_ready = threading.Event()
        self._writer_ready = threading.Event()
        self._started = False
        self._finished = False
        self._next_submit = 0
        self._frames_delivered = 0
        self._chunk_records: Dict[int, Dict[str, float]] = {}
        self._records_lock = threading.Lock()
        self._sink_result: Any = None
        self._timeline: Dict[str, Any] = {}
        self._writer_frame_cursor = 0
        #: Workers still alive after a failed session's join timeout ("decoder" / "writer").
        self.stuck_workers: Tuple[str, ...] = ()

    # ------------------------------------------------------------------ utils
    def _mark(self, name: str, **extra: Any) -> float:
        if self.marks is not None:
            return self.marks.mark(name, **extra)
        return time.monotonic()

    def _record(self, idx: int, **kv: float) -> None:
        with self._records_lock:
            self._chunk_records.setdefault(idx, {"chunk_index": idx}).update(kv)

    # ------------------------------------------------------------------ lifecycle
    def start(self, ready_timeout: float = 120.0) -> None:
        assert not self._started, "session already started"
        self._started = True
        self._timeline["start"] = time.monotonic()
        t0 = time.monotonic()
        try:
            self._writer_thread = threading.Thread(target=self._writer_main, name="omnimem-writer", daemon=True)
            self._writer_thread.start()
            if self.decode_in_caller:
                self.decoder.start()
                self._decoder_ready.set()
            else:
                self._decoder_thread = threading.Thread(target=self._decoder_main, name="omnimem-decoder", daemon=True)
                self._decoder_thread.start()
            while not (self._decoder_ready.is_set() and self._writer_ready.is_set()):
                self._failure.check()
                if time.monotonic() - t0 > ready_timeout:
                    raise TimeoutError("workers did not report READY")
                time.sleep(0.01)
        except BaseException as exc:
            if not self._failure.failed:
                self._failure.fail("start", exc)
            self._abort_workers()
            self._failure.check()
        self._timeline["workers_ready"] = time.monotonic()

    def submit(self, chunk: LatentChunk) -> None:
        """Called on the producer thread for every final latent chunk, in order."""
        try:
            self._failure.check()
            if chunk.chunk_index != self._next_submit:
                raise ValueError(f"chunk {chunk.chunk_index} submitted, expected {self._next_submit}")
            self._next_submit += 1
            self._record(chunk.chunk_index, submit=chunk.submit_time)
            if self.decode_in_caller:
                self._decode_one(chunk, torch.cuda.current_stream(self.decode_device), None)
            else:
                self._latent_q.put(chunk, self.put_timeout)
        except BaseException as exc:
            if not self._failure.failed:
                self._failure.fail("submit", exc)
            self._abort_workers()
            if isinstance(exc, SessionError):
                raise
            self._failure.check()

    def finish(self) -> SessionResult:
        """Send EOS, drain both workers, finalise the sink.  Raises SessionError on failure."""
        assert self._started and not self._finished
        self._finished = True
        try:
            self._timeline["producer_done"] = time.monotonic()
            if self.decode_in_caller:
                try:
                    self.decoder.finish()
                finally:
                    self._frame_q.put(EOS, self.put_timeout)
            else:
                self._latent_q.put(EOS, self.put_timeout)
                self._join(self._decoder_thread, "decoder")
            self._timeline["decoder_drained"] = time.monotonic()
            self._join(self._writer_thread, "writer")
            self._timeline["writer_done"] = time.monotonic()
            self._failure.check()
        except BaseException as exc:
            if not self._failure.failed:
                self._failure.fail("finish", exc)
            self._abort_workers()
            raise
        records = [self._chunk_records[k] for k in sorted(self._chunk_records)]
        return SessionResult(
            frames_delivered=self._frames_delivered,
            sink_result=self._sink_result,
            timeline=dict(self._timeline),
            chunk_records=records,
            queue_stats={"latent_queue": self._latent_q.stats(), "frame_queue": self._frame_q.stats(),
                         "d2h_pool": {"acquire_blocked_s": self._pool.acquire_blocked_s}},
            d2h_pool_bytes=self._pool.bytes,
        )

    def abort(self, exc: Optional[BaseException] = None) -> None:
        """Cancel from the producer side (e.g. the sampler raised)."""
        self._failure.fail("producer", exc if exc is not None else RuntimeError("aborted by producer"))
        self._abort_workers()

    def _abort_sink(self) -> None:
        try:
            self.sink.abort()
        except Exception:
            pass

    def _abort_workers(self) -> None:
        """Stop the workers and release the sink (aborted first so a stalled writer wakes)."""
        self._abort_sink()
        deadline = time.monotonic() + self.join_timeout
        stuck = []
        for name, th in (("decoder", self._decoder_thread), ("writer", self._writer_thread)):
            if th is None:
                continue
            th.join(timeout=max(0.0, deadline - time.monotonic()))
            if th.is_alive():
                stuck.append(name)
        self.stuck_workers = tuple(stuck)
        if self.decode_in_caller:
            try:
                self.decoder.finish()
            except Exception:
                pass
        # Again: a writer that was between two sink calls may have opened the encoder
        # after the first abort.
        self._abort_sink()

    def _join(self, th: Optional[threading.Thread], name: str) -> None:
        if th is None:
            return
        t0 = time.monotonic()
        while th.is_alive():
            th.join(timeout=0.1)
            if self._failure.failed:
                th.join(timeout=self.join_timeout)
                break
            if time.monotonic() - t0 > self.join_timeout:
                raise TimeoutError(f"{name} thread did not finish within {self.join_timeout:.0f}s")

    # ------------------------------------------------------------------ decoder
    def _decode_one(self, chunk: LatentChunk, decode_stream, transfer_stream) -> None:
        """Transfer (if needed), decode, convert and D2H one chunk; enqueue for the writer."""
        idx = chunk.chunk_index
        latent = chunk.latent
        if latent.device != self.dit_device:
            raise ValueError(f"chunk {idx} latent lives on {latent.device}, the sampler "
                             f"device of this session is {self.dit_device}")
        if self.decode_in_caller:
            # One device: decoded on the calling thread, on the stream that produced
            # the snapshot.
            if chunk.ready_event is not None:
                decode_stream.wait_event(chunk.ready_event)
            z = latent
            self._record(idx, gpu_ready=time.monotonic())
        else:
            if chunk.ready_event is not None:
                transfer_stream.wait_event(chunk.ready_event)
            z = torch.empty(latent.shape, dtype=latent.dtype, device=self.decode_device)
            # Copy on the source stream; keep the source referenced until the copy event completes.
            z.copy_(latent, non_blocking=True)
            copy_done = torch.cuda.Event()
            copy_done.record(transfer_stream)
            copy_done.synchronize()
            t_copy = self._mark("latent_copy_done", chunk=idx)
            self._record(idx, gpu_ready=t_copy, copy_done=t_copy)
        chunk.latent = None  # drop producer-side reference as early as possible
        del latent

        first = idx == 0
        n_out = output_frames_for_latents(z.shape[2], first_chunk=first)
        out = self.decoder.decode(z)
        if out.shape[2] != n_out:
            raise RuntimeError(f"decoder produced {out.shape[2]} frames for chunk {idx}, expected {n_out}")
        u8 = rgb_uint8_from_decoded(out)
        del out, z
        if tuple(u8.shape[1:3]) != self.frame_hw:
            raise RuntimeError(f"decoded frame size {tuple(u8.shape[1:3])} != expected {self.frame_hw}")
        if u8.shape[0] > self.max_frames_per_chunk:
            raise RuntimeError(f"chunk {idx} decoded to {u8.shape[0]} frames > pool capacity {self.max_frames_per_chunk}")
        decode_done_ev = torch.cuda.Event()
        decode_done_ev.record(decode_stream)

        slot = self._pool.acquire(self.get_timeout)
        dst = slot[:u8.shape[0]]
        dst.copy_(u8, non_blocking=True)
        d2h_ev = torch.cuda.Event()
        d2h_ev.record(decode_stream)
        decode_done_ev.synchronize()
        t_dec = self._mark("decode_done", chunk=idx, frames=int(u8.shape[0]))
        d2h_ev.synchronize()
        t_d2h = self._mark("d2h_done", chunk=idx)
        del u8
        self._record(idx, decode_done=t_dec, d2h_done=t_d2h, frames=float(dst.shape[0]))
        pool = self._pool
        item = DecodedChunk(chunk_index=idx, frames=dst,
                            release=lambda s=slot: pool.release(s))
        self._frame_q.put(item, self.put_timeout)

    def _decoder_main(self) -> None:
        decode_stream = transfer_stream = None
        try:
            torch.cuda.set_device(self.decode_device)
            decode_stream, transfer_stream = (
                self.decoder_streams if self.decoder_streams is not None
                else decoder_thread_streams(self.decode_device, self.dit_device))
            self.decoder.start()
            self._decoder_ready.set()
            expected = 0
            # Enter the transfer stream first so the decode stream leaves the decode device current.
            with torch.inference_mode(), torch.cuda.stream(transfer_stream), \
                    torch.cuda.stream(decode_stream), \
                    torch.autocast("cuda", dtype=self.autocast_dtype):
                torch.cuda.set_device(self.decode_device)
                while True:
                    item = self._latent_q.get(self.get_timeout)
                    if item is EOS:
                        break
                    if item.chunk_index != expected:
                        raise ValueError(f"decoder received chunk {item.chunk_index}, expected {expected}")
                    self._decode_one(item, decode_stream, transfer_stream)
                    expected += 1
                decode_stream.synchronize()
            self._frame_q.put(EOS, self.put_timeout)
        except BaseException as exc:  # noqa: BLE001 - must reach the shared state
            self._failure.fail("decoder", exc)
        finally:
            try:
                if decode_stream is not None:
                    decode_stream.synchronize()
                if transfer_stream is not None:
                    transfer_stream.synchronize()
            except Exception:
                pass
            try:
                self.decoder.finish()
            except Exception as exc:  # noqa: BLE001
                self._failure.fail("decoder.finish", exc)

    # ------------------------------------------------------------------ writer
    def _writer_main(self) -> None:
        try:
            self._writer_ready.set()
            while True:
                item = self._frame_q.get(self.get_timeout)
                if item is EOS:
                    break
                n = int(item.frames.shape[0])
                frame_start = self._writer_frame_cursor
                self.sink.on_frames(item.frames, frame_start=frame_start, chunk_index=item.chunk_index)
                t = self._mark("frames_delivered", chunk=item.chunk_index, frames=n, frame_start=frame_start)
                self._writer_frame_cursor += n
                self._frames_delivered += n
                self._record(item.chunk_index, delivered=t)
                if item.release is not None:
                    item.release()
            self._sink_result = self.sink.finish()
            self._mark("sink_finished")
        except BaseException as exc:  # noqa: BLE001
            self._failure.fail("writer", exc)
            try:
                self.sink.abort()
            except Exception:
                pass
