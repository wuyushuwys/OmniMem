"""Incremental MP4 writing: one ffmpeg process per file, fed raw RGB24 frames."""
from __future__ import annotations

import os
import subprocess
import threading
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np
import PIL.Image

try:
    import torch
except ImportError:  # pragma: no cover - torch is a hard dependency of this repo
    torch = None  # type: ignore[assignment]

FrameInput = Union[np.ndarray, "torch.Tensor"]


class StreamingWriterError(RuntimeError):
    """Raised when the ffmpeg subprocess fails to start, write, or finalize a video."""


@dataclass
class WriterResult:
    """Outcome of a successful :meth:`StreamingVideoWriter.finalize` call."""

    path: str
    frames_written: int
    seconds: float
    ffmpeg_returncode: int


def save_frame_png(frame_u8_hw3: np.ndarray, path: str) -> None:
    """Save a single ``[H, W, 3]`` uint8 RGB frame as a PNG at ``path``."""
    if frame_u8_hw3.ndim != 3 or frame_u8_hw3.shape[2] != 3 or frame_u8_hw3.dtype != np.uint8:
        raise ValueError(
            f"expected a uint8 [H, W, 3] array, got shape={frame_u8_hw3.shape} "
            f"dtype={frame_u8_hw3.dtype}"
        )
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    PIL.Image.fromarray(frame_u8_hw3).save(path)


def probe_video_frames(path: str, ffprobe: str = "ffprobe") -> int:
    """Return the number of video frames (packets) in the video at ``path``."""
    cmd = [
        ffprobe, "-v", "error", "-select_streams", "v:0", "-count_packets",
        "-show_entries", "stream=nb_read_packets", "-of", "csv=p=0", path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        raise StreamingWriterError(f"ffprobe failed to run: {e}") from e
    if out.returncode != 0:
        raise StreamingWriterError(f"ffprobe exited with {out.returncode}: {out.stderr.strip()}")
    text = out.stdout.strip()
    try:
        return int(text)
    except ValueError as e:
        raise StreamingWriterError(f"ffprobe returned non-integer frame count: {text!r}") from e


def _drain_stderr(pipe, buf: List[bytes], lock: threading.Lock, keep_bytes: int) -> None:
    """Daemon-thread target: read ffmpeg stderr as it arrives, keeping only the tail."""
    try:
        for chunk in iter(lambda: pipe.read(4096), b""):
            with lock:
                buf.append(chunk)
                total = sum(len(b) for b in buf)
                while total > keep_bytes and len(buf) > 1:
                    total -= len(buf.pop(0))
    except (OSError, ValueError):
        pass
    finally:
        try:
            pipe.close()
        except OSError:
            pass


class StreamingVideoWriter:
    """Streams raw RGB24 frames into one ffmpeg process to build an MP4; not thread-safe."""

    _STDERR_KEEP_BYTES = 4096

    def __init__(
        self,
        output_path: str,
        width: int,
        height: int,
        fps: float,
        *,
        crf: int = 18,
        preset: str = "medium",
        pix_fmt: str = "yuv420p",
        codec: str = "libx264",
        fragmented: bool = False,
        ffmpeg: str = "ffmpeg",
        preview_dir: Optional[str] = None,
        preview_every_frames: Optional[int] = None,
        extra_ffmpeg_args: Optional[List[str]] = None,
    ) -> None:
        if width % 2 != 0 or height % 2 != 0:
            raise ValueError(
                f"width and height must be even for {codec}/{pix_fmt} (got {width}x{height})"
            )
        self.output_path = output_path
        self.width = width
        self.height = height
        self.fps = fps
        self.crf = crf
        self.preset = preset
        self.pix_fmt = pix_fmt
        self.codec = codec
        self.fragmented = fragmented
        self.ffmpeg = ffmpeg
        self.preview_dir = preview_dir
        self.preview_every_frames = preview_every_frames
        self.extra_ffmpeg_args = list(extra_ffmpeg_args) if extra_ffmpeg_args else []

        self._temp_path = os.path.join(
            os.path.dirname(output_path) or ".",
            f"{os.path.basename(output_path)}.partial.{os.getpid()}.mp4",
        )
        self.command: List[str] = self._build_command()

        self._proc: Optional[subprocess.Popen] = None
        self._stderr_buf: List[bytes] = []
        self._stderr_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._is_open = False
        self._failed = False
        self._start_time: Optional[float] = None

        self._frames_written = 0
        self.preview_paths: List[str] = []
        self.first_frame_time: Optional[float] = None
        self.first_write_time: Optional[float] = None
        self.batch_times: List[Tuple[float, int]] = []

    def _build_command(self) -> List[str]:
        cmd = [
            self.ffmpeg, "-y", "-hide_banner", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{self.width}x{self.height}", "-r", str(self.fps), "-i", "-",
            "-an", "-c:v", self.codec, "-preset", self.preset, "-crf", str(self.crf),
            "-pix_fmt", self.pix_fmt,
        ]
        if self.fragmented:
            cmd += ["-movflags", "+frag_keyframe+empty_moov+default_base_moof"]
        cmd += self.extra_ffmpeg_args
        cmd += [self._temp_path]
        return cmd

    @property
    def frames_written(self) -> int:
        return self._frames_written

    @property
    def temp_path(self) -> str:
        return self._temp_path

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        """Create the parent directory and spawn the ffmpeg subprocess."""
        if self._is_open:
            raise StreamingWriterError("writer is already open")
        os.makedirs(os.path.dirname(self.output_path) or ".", exist_ok=True)
        try:
            self._proc = subprocess.Popen(
                self.command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except OSError as e:
            raise StreamingWriterError(f"failed to spawn ffmpeg: {e}") from e
        self._stderr_thread = threading.Thread(
            target=_drain_stderr,
            args=(self._proc.stderr, self._stderr_buf, self._stderr_lock, self._STDERR_KEEP_BYTES),
            daemon=True,
        )
        self._stderr_thread.start()
        self._is_open = True
        self._failed = False
        self._start_time = time.monotonic()

    def _stderr_tail(self) -> str:
        with self._stderr_lock:
            data = b"".join(self._stderr_buf)
        return data[-self._STDERR_KEEP_BYTES:].decode("utf-8", errors="replace")

    def _to_numpy_frames(self, frames: FrameInput) -> np.ndarray:
        if torch is not None and isinstance(frames, torch.Tensor):
            if frames.is_cuda:
                raise ValueError("frames tensor must be on CPU, got a CUDA tensor")
            frames = frames.numpy()
        if not isinstance(frames, np.ndarray):
            raise ValueError(f"frames must be a numpy array or CPU torch tensor, got {type(frames)}")
        if frames.dtype != np.uint8:
            raise ValueError(f"frames must be uint8, got dtype={frames.dtype}")
        if frames.ndim == 3:
            frames = frames[None]
        if frames.ndim != 4 or frames.shape[1:] != (self.height, self.width, 3):
            raise ValueError(
                f"frames must have shape [T, {self.height}, {self.width}, 3] or "
                f"[{self.height}, {self.width}, 3], got {frames.shape}"
            )
        if not frames.flags["C_CONTIGUOUS"]:
            frames = np.ascontiguousarray(frames)
        return frames

    def write_frames(self, frames: FrameInput) -> int:
        """Write frames (``[T, H, W, 3]`` or ``[H, W, 3]``, uint8) to ffmpeg; return the count."""
        if self._failed:
            raise StreamingWriterError(f"writer already failed: {self._stderr_tail()}")
        if not self._is_open or self._proc is None:
            raise StreamingWriterError("writer is not open; call open() first")

        np_frames = self._to_numpy_frames(frames)
        n = np_frames.shape[0]

        try:
            self._proc.stdin.write(memoryview(np_frames))
        except (BrokenPipeError, OSError) as e:
            self._failed = True
            tail = self._stderr_tail()
            raise StreamingWriterError(f"ffmpeg pipe write failed: {e}. stderr tail:\n{tail}") from e

        start_index = self._frames_written
        self._frames_written += n
        now = time.monotonic()
        if self.first_write_time is None:
            self.first_write_time = now
        self.batch_times.append((now, n))

        if self.preview_dir is not None:
            self._maybe_save_previews(np_frames, start_index)

        return self._frames_written

    def _maybe_save_previews(self, np_frames: np.ndarray, start_index: int) -> None:
        for i in range(np_frames.shape[0]):
            abs_index = start_index + i
            take = abs_index == 0
            if not take and self.preview_every_frames:
                take = abs_index % self.preview_every_frames == 0
            if not take:
                continue
            path = os.path.join(self.preview_dir, f"frame_{abs_index:06d}.png")
            save_frame_png(np_frames[i], path)
            self.preview_paths.append(path)
            if abs_index == 0:
                self.first_frame_time = time.monotonic()

    def flush(self) -> None:
        """Flush ffmpeg's stdin so written frames reach disk (useful with ``fragmented=True``)."""
        if not self._is_open or self._proc is None or self._proc.stdin is None:
            raise StreamingWriterError("writer is not open; call open() first")
        try:
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError) as e:
            self._failed = True
            tail = self._stderr_tail()
            raise StreamingWriterError(f"ffmpeg pipe flush failed: {e}. stderr tail:\n{tail}") from e

    def finalize(self, timeout: float = 600.0) -> WriterResult:
        """Close stdin, wait for ffmpeg and publish the temp file to ``output_path``."""
        if not self._is_open or self._proc is None:
            raise StreamingWriterError("writer is not open; call open() first")
        if self._failed:
            self._cleanup_temp()
            self._is_open = False
            raise StreamingWriterError(f"writer already failed: {self._stderr_tail()}")

        try:
            if self._proc.stdin is not None:
                self._proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass

        try:
            returncode = self._proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired as e:
            self._proc.kill()
            self._proc.wait()
            self._is_open = False
            self._join_stderr_thread()
            self._cleanup_temp()
            raise StreamingWriterError(f"ffmpeg timed out after {timeout}s") from e

        self._join_stderr_thread()
        self._is_open = False

        if returncode != 0:
            tail = self._stderr_tail()
            self._cleanup_temp()
            raise StreamingWriterError(f"ffmpeg exited with code {returncode}. stderr tail:\n{tail}")

        os.replace(self._temp_path, self.output_path)
        elapsed = time.monotonic() - (self._start_time or time.monotonic())
        return WriterResult(
            path=self.output_path,
            frames_written=self._frames_written,
            seconds=elapsed,
            ffmpeg_returncode=returncode,
        )

    def abort(self) -> None:
        """Kill ffmpeg if still running and remove the temp file. Never renames. Idempotent."""
        if self._proc is not None and self._proc.poll() is None:
            try:
                self._proc.kill()
                self._proc.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self._join_stderr_thread()
        self._is_open = False
        self._cleanup_temp()

    def _join_stderr_thread(self) -> None:
        # Swap before joining: ``abort`` may be called from a second thread while the
        # writer thread is finalising.
        thread, self._stderr_thread = self._stderr_thread, None
        if thread is not None:
            thread.join(timeout=10)

    def _cleanup_temp(self) -> None:
        try:
            os.remove(self._temp_path)
        except OSError:
            pass

    def __enter__(self) -> "StreamingVideoWriter":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if exc_type is None:
            self.finalize()
        else:
            self.abort()
