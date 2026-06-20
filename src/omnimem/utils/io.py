import os
import warnings
from typing import List

import PIL
import PIL.Image
import PIL.ImageOps
import numpy as np
import imageio

import torch
import torchvision
import torchvision.transforms.functional
from torchvision.io import write_video
from einops import rearrange

from .logging_tool import get_logger

warnings.filterwarnings("ignore")

logger = get_logger()


def frames_to_gif(frames, output_file, fps=7, generic=True, release=False):

    if len(frames) == 1 and generic:
        frame = PIL.Image.fromarray(frames[0])
        extension = os.path.splitext(output_file)[-1]
        output_file = output_file.replace(extension, ".jpg")
        frame.save(output_file)
        return output_file
    imageio.mimsave(output_file, frames, fps=fps, loop=0)
    return output_file


def frames_to_video_tv(frames, output_file: str, fps: float = 24):
    frames = [torch.from_numpy(f) for f in frames]
    write_video(
        filename=output_file,
        video_array=torch.stack(frames, dim=0),
        fps=fps,
        video_codec="libx264",  # widely compatible H.264 encoder
        options={"crf": "18", "pix_fmt": "yuv420p"},  # good quality + broad playback support
    )
    return output_file


def save_videos_grid_pil(videos: List[List[PIL.Image.Image]], path: str, n_rows=6, fps=8, verbose=False, generic=True,
                         release=False):
    video_grid = []

    for frames in videos:
        video_grid.append(torch.stack([torchvision.transforms.functional.to_tensor(frame) for frame in frames]))
    videos = torch.stack(video_grid)
    videos = rearrange(videos, "b t c h w -> t b c h w")
    outputs = []
    for x in videos:
        if n_rows > 1:
            x = torchvision.utils.make_grid(x, nrow=n_rows)
        if x.ndim == 4:
            x = x.squeeze(0)
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    if path.endswith('.gif'):
        output_file = frames_to_gif(outputs, path, fps=fps, generic=generic, release=release)
    else:
        output_file = frames_to_video_tv(outputs, path, fps=fps)
    return output_file


