import copy
import gc
import math
from io import BytesIO
import os
import re
import warnings
from fractions import Fraction
from typing import Any, Dict, List, Optional, Tuple, Union

import decord

decord.bridge.set_bridge('torch')

import av
import cv2
import numpy as np
import torch
from torchvision import get_video_backend
from torchvision.io.video import _check_av_available

MAX_NUM_FRAMES = 2500


def read_video_av(
        filename: Union[str, BytesIO],
        start_pts: Union[float, Fraction] = 0,
        end_pts: Optional[Union[float, Fraction]] = None,
        pts_unit: str = "pts",
        output_format: str = "THWC",
) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
    """Read video via pyav; no audio, no memory leaks. Modified from torchvision.io.video.read_video.

    Args:
        filename: path or BytesIO to video file.
        start_pts: start presentation time.
        end_pts: end presentation time.
        pts_unit: unit for pts values ('pts' or 'sec').
        output_format: output tensor layout ('THWC' or 'TCHW').

    Returns:
        vframes, aframes (empty), info dict.
    """
    filename_is_str = isinstance(filename, str)
    filename_copy = None
    if not filename_is_str:
        filename_copy = copy.deepcopy(filename)
    output_format = output_format.upper()
    if output_format not in ("THWC", "TCHW"):
        raise ValueError(f"output_format should be either 'THWC' or 'TCHW', got {output_format}.")
    if filename_is_str and not os.path.exists(filename):
        raise RuntimeError(f"File not found: {filename}")
    assert get_video_backend() == "pyav", "pyav backend is required for read_video_av"
    _check_av_available()
    if end_pts is None:
        end_pts = float("inf")
    if end_pts < start_pts:
        raise ValueError(f"end_pts should be larger than start_pts, got start_pts={start_pts} and end_pts={end_pts}")

    # get video info
    info = {}
    container = av.open(filename, metadata_errors="ignore")
    video_fps = container.streams.video[0].average_rate
    # guard against potentially corrupted files
    if video_fps is not None:
        info["video_fps"] = float(video_fps)
    iter_video = container.decode(**{"video": 0})
    frame = next(iter_video).to_rgb().to_ndarray()
    height, width = frame.shape[:2]
    total_frames = container.streams.video[0].frames
    if total_frames == 0:
        total_frames = MAX_NUM_FRAMES
        warnings.warn(f"total_frames is 0, using {MAX_NUM_FRAMES} as a fallback")
    container.close()
    del container

    video_frames = np.zeros((total_frames, height, width, 3), dtype=np.uint8)

    # read frames
    try:
        if filename_is_str:
            container = av.open(filename, metadata_errors="ignore")
        else:
            container = av.open(filename_copy, metadata_errors="ignore")
        assert container.streams.video is not None
        video_frames = _read_from_stream(
            video_frames,
            container,
            start_pts,
            end_pts,
            pts_unit,
            container.streams.video[0],
            {"video": 0},
            filename=filename,
        )
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")

    vframes = torch.from_numpy(video_frames).clone()
    del video_frames
    if output_format == "TCHW":  # [T,H,W,C] -> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)

    aframes = torch.empty((1, 0), dtype=torch.float32)
    return vframes, aframes, info


def _read_from_stream(
        video_frames,
        container: "av.container.Container",
        start_offset: float,
        end_offset: float,
        pts_unit: str,
        stream: "av.stream.Stream",
        stream_name: Dict[str, Optional[Union[int, Tuple[int, ...], List[int]]]],
        filename: Optional[str] = None,
) -> List["av.frame.Frame"]:
    if pts_unit == "sec":
        start_offset = int(math.floor(start_offset * (1 / stream.time_base)))
        if end_offset != float("inf"):
            end_offset = int(math.ceil(end_offset * (1 / stream.time_base)))
    else:
        warnings.warn("The pts_unit 'pts' gives wrong results. Please use pts_unit 'sec'.")

    should_buffer = True
    max_buffer_size = 5
    if stream.type == "video":
        # DivX packed B-frames can have out-of-order pts; buffer to sort correctly
        extradata = stream.codec_context.extradata
        if extradata and b"DivX" in extradata:
            # can't use regex directly because of some weird characters sometimes...
            pos = extradata.find(b"DivX")
            d = extradata[pos:]
            o = re.search(rb"DivX(\d+)Build(\d+)(\w)", d)
            if o is None:
                o = re.search(rb"DivX(\d+)b(\d+)(\w)", d)
            if o is not None:
                should_buffer = o.group(3) == b"p"
    seek_offset = start_offset
    seek_offset = max(seek_offset - 1, 0)
    if should_buffer:
        seek_offset = max(seek_offset - max_buffer_size, 0)
    try:
        container.seek(seek_offset, any_frame=False, backward=True, stream=stream)
    except av.AVError as e:
        print(f"[Warning] Error while seeking video {filename}: {e}")
        return []

    # decode frames
    buffer_count = 0
    frames_pts = []
    cnt = 0
    try:
        for _idx, frame in enumerate(container.decode(**stream_name)):
            frames_pts.append(frame.pts)
            video_frames[cnt] = frame.to_rgb().to_ndarray()
            cnt += 1
            if cnt >= len(video_frames):
                break
            if frame.pts >= end_offset:
                if should_buffer and buffer_count < max_buffer_size:
                    buffer_count += 1
                    continue
                break
    except av.AVError as e:
        print(f"[Warning] Error while reading video {filename}: {e}")

    container.close()
    del container
    gc.collect()

    # frames_pts is assumed sorted
    start_ptr = 0
    end_ptr = cnt
    while start_ptr < end_ptr and frames_pts[start_ptr] < start_offset:
        start_ptr += 1
    while start_ptr < end_ptr and frames_pts[end_ptr - 1] > end_offset:
        end_ptr -= 1
    if start_offset > 0 and start_offset not in frames_pts[start_ptr:end_ptr]:
        # if there is no frame that exactly matches the pts of start_offset
        # add the last frame smaller than start_offset, to guarantee that
        # we will have all the necessary data. This is most useful for audio
        if start_ptr > 0:
            start_ptr -= 1
    result = video_frames[start_ptr:end_ptr].copy()
    return result


def read_video_cv2(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError
    else:
        fps = cap.get(cv2.CAP_PROP_FPS)
        vinfo = {
            "video_fps": fps,
        }

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frames.append(frame[:, :, ::-1])  # BGR to RGB
            if cv2.waitKey(25) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()

        frames = np.stack(frames)
        frames = torch.from_numpy(frames)  # [T, H, W, C=3]
        frames = frames.permute(0, 3, 1, 2)
        return frames, vinfo


def read_video_decord(
        filename: Union[str, BytesIO],
        frame_index: Optional[Union[List[int], str, np.ndarray]] = 'all',
        output_format: str = "THWC",
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    vr = decord.VideoReader(filename)
    vr.seek(0)
    vinfo = dict(
        key_indices=vr.get_key_indices()
    )
    video_fps = vr.get_avg_fps()
    if video_fps is not None:
        vinfo["video_fps"] = float(video_fps)

    if frame_index is None:
        vframes = vr
    elif isinstance(frame_index, str) and frame_index == "all":
        vframes = vr.get_batch(range(len(vr)))
    else:
        if max(frame_index) >= len(vr):
            if max(frame_index) - min(frame_index) <= len(vr):
                offset = max(frame_index) - len(vr) + 1
                frame_index -= offset
                assert min(frame_index) >= 0, f"frame_index {frame_index} exceeds the length of video {len(vr)}"
            else:
                raise ValueError(f"frame_index {frame_index} exceeds the length of video {len(vr)}")
        vframes = vr.get_batch(frame_index)
    vr.seek(0)
    if output_format == "TCHW" and frame_index is not None:  # [T,H,W,C] -> [T,C,H,W]
        vframes = vframes.permute(0, 3, 1, 2)
    return vframes, vinfo


def read_video(video_path, backend="decord", frame_index: Union[str, np.ndarray]='all'):
    if backend == "cv2":
        vframes, vinfo = read_video_cv2(video_path)
    elif backend == "av":
        vframes, _, vinfo = read_video_av(filename=video_path, pts_unit="sec", output_format="TCHW")
        if not isinstance(frame_index, str):
            vframes = vframes[frame_index]
    elif backend == "decord":
        vframes, vinfo = read_video_decord(filename=video_path, frame_index=frame_index, output_format="TCHW")
    else:
        raise ValueError

    return vframes, vinfo
