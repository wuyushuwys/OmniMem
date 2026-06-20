import numbers
import random
from typing import List, Union
import torch
from torchvision.transforms.v2 import functional as F


""" torchvision.transform.v2 """


class ResizeCropToFillV2(torch.nn.Module):
    """Resize-to-fill and center crop using torchvision.transforms.v2."""

    def __init__(self, size: Union[List[int], int], interpolation=F.InterpolationMode.BICUBIC, antialias=True):
        super().__init__()
        self.interpolation = interpolation
        self.antialias = antialias
        self.size = size

    def forward(self, media):
        # get original size; F.get_size handles both Tensors and PIL Images
        original_size = F.get_size(media)
        original_h, original_w = original_size[0], original_size[1]

        target_h, target_w = self.size

        # scale by the larger ratio to cover the target area
        ratio_h = target_h / original_h
        ratio_w = target_w / original_w

        if ratio_h > ratio_w:
            # scale based on height
            new_h = target_h
            new_w = int(original_w * ratio_h)
        else:
            # scale based on width
            new_h = int(original_h * ratio_w)
            new_w = target_w

        # resize
        resized_media = F.resize(
            media,
            size=[new_h, new_w],
            interpolation=self.interpolation,
            antialias=self.antialias,
        )

        # center crop
        cropped_media = F.center_crop(resized_media, output_size=self.size)

        return cropped_media

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation={self.interpolation})"


def _is_tensor_video_clip(clip):
    if not torch.is_tensor(clip):
        raise TypeError("clip should be Tensor. Got %s" % type(clip))

    if not clip.ndimension() == 4:
        raise ValueError("clip should be 4D. Got %dD" % clip.dim())

    return True


def crop(clip, i, j, h, w):
    """
    Args:
        clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
    """
    if len(clip.size()) != 4:
        raise ValueError("clip should be a 4D tensor")
    return clip[..., i: i + h, j: j + w]


def resize(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    return torch.nn.functional.interpolate(clip,
                                           size=target_size,
                                           mode=interpolation_mode,
                                           antialias=True,
                                           align_corners=False)


def resize_scale(clip, target_size, interpolation_mode):
    if len(target_size) != 2:
        raise ValueError(f"target size should be tuple (height, width), instead got {target_size}")
    H, W = clip.size(-2), clip.size(-1)
    scale_ = target_size[0] / min(H, W)
    return torch.nn.functional.interpolate(clip, scale_factor=scale_, mode=interpolation_mode,
                                           antialias=True,
                                           align_corners=False)


def center_crop(clip, crop_size):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = crop_size
    if h < th or w < tw:
        raise ValueError("height and width must be no smaller than crop_size")

    i = int(round((h - th) / 2.0))
    j = int(round((w - tw) / 2.0))
    return crop(clip, i, j, th, tw)


def center_crop_using_short_edge(clip):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    if h < w:
        th, tw = h, h
        i = 0
        j = int(round((w - tw) / 2.0))
    else:
        th, tw = w, w
        i = int(round((h - th) / 2.0))
        j = 0
    return crop(clip, i, j, th, tw)


def resize_crop_to_fill(clip, target_size, mode="bilinear"):
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)
    th, tw = target_size[0], target_size[1]
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        clip = resize(clip, (sh, sw), mode)
        i = 0
        j = int(round(sw - tw) / 2.0)
    else:
        sh, sw = round(h * rw), tw
        clip = resize(clip, (sh, sw), mode)
        i = int(round(sh - th) / 2.0)
        j = 0
    assert i + th <= clip.size(-2) and j + tw <= clip.size(-1)
    return crop(clip, i, j, th, tw)


def random_shift_crop(clip):
    """
    Slide along the long edge, with the short edge as crop size
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    h, w = clip.size(-2), clip.size(-1)

    if h <= w:
        short_edge = h
    else:
        short_edge = w

    th, tw = short_edge, short_edge

    i = torch.randint(0, h - th + 1, size=(1,)).item()
    j = torch.randint(0, w - tw + 1, size=(1,)).item()
    return crop(clip, i, j, th, tw)


def to_tensor(clip):
    """Convert uint8 tensor (T,C,H,W) to float32 by dividing by 255."""
    _is_tensor_video_clip(clip)
    if not clip.dtype == torch.uint8:
        raise TypeError("clip tensor should have data type uint8. Got %s" % str(clip.dtype))
    return clip.float() / 255.0


def normalize(clip, mean, std, inplace=False):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
        mean (tuple): pixel RGB mean. Size is (3)
        std (tuple): pixel standard deviation. Size is (3)
    Returns:
        normalized clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    if not inplace:
        clip = clip.clone()
    mean = torch.as_tensor(mean, dtype=clip.dtype, device=clip.device)
    std = torch.as_tensor(std, dtype=clip.dtype, device=clip.device)
    clip.sub_(mean[:, None, None, None]).div_(std[:, None, None, None])
    return clip


def hflip(clip):
    """
    Args:
        clip (torch.tensor): Video clip to be normalized. Size is (T, C, H, W)
    Returns:
        flipped clip (torch.tensor): Size is (T, C, H, W)
    """
    if not _is_tensor_video_clip(clip):
        raise ValueError("clip should be a 4D torch.tensor")
    return clip.flip(-1)


class ResizeCrop:
    def __init__(self, size, mode='bicubic'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.mode = mode

    def __call__(self, clip):
        clip = resize_crop_to_fill(clip, self.size, self.mode)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class RandomCropVideo:
    def __init__(self, size, mode='bicubic'):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.mode = mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: randomly cropped video clip.
                size is (T, C, OH, OW)
        """
        try:
            i, j, h, w = self.get_params(clip)
            return crop(clip, i, j, h, w)
        except ValueError:
            return resize_crop_to_fill(clip, self.size, self.mode)

    def get_params(self, clip):
        h, w = clip.shape[-2:]
        th, tw = self.size

        if h < th or w < tw:
            raise ValueError(f"Required crop size {(th, tw)} is larger than input image size {(h, w)}")

        if w == tw and h == th:
            return 0, 0, h, w

        i = torch.randint(0, h - th + 1, size=(1,)).item()
        j = torch.randint(0, w - tw + 1, size=(1,)).item()

        return i, j, th, tw

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size})"


class CenterCropResizeVideo:
    """
    First use the short side for cropping length,
    center crop video, then resize to the specified size
    """

    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop_using_short_edge(clip)
        clip_center_crop_resize = resize(
            clip_center_crop, target_size=self.size, interpolation_mode=self.interpolation_mode
        )
        return clip_center_crop_resize

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class UCFCenterCropVideo:
    """
    First scale to the specified size in equal proportion to the short edge,
    then center cropping
    """

    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: scale resized / center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_resize = resize_scale(clip=clip, target_size=self.size, interpolation_mode=self.interpolation_mode)
        clip_center_crop = center_crop(clip_resize, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class KineticsRandomCropResizeVideo:
    """
    Slide along the long edge, with the short edge as crop size. And resie to the desired size.
    """

    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        clip_random_crop = random_shift_crop(clip)
        clip_resize = resize(clip_random_crop, self.size, self.interpolation_mode)
        return clip_resize


class CenterCropVideo:
    def __init__(
            self,
            size,
            interpolation_mode="bilinear",
    ):
        if isinstance(size, tuple):
            if len(size) != 2:
                raise ValueError(f"size should be tuple (height, width), instead got {size}")
            self.size = size
        else:
            self.size = (size, size)

        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (T, C, H, W)
        Returns:
            torch.tensor: center cropped video clip.
                size is (T, C, crop_size, crop_size)
        """
        clip_center_crop = center_crop(clip, self.size)
        return clip_center_crop

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(size={self.size}, interpolation_mode={self.interpolation_mode}"


class NormalizeVideo:
    """Normalize video clip by mean subtraction and std division.

    Args:
        mean: pixel RGB mean.
        std: pixel RGB standard deviation.
        inplace: whether to normalize in-place.
    """

    def __init__(self, mean, std, inplace=False):
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): video clip must be normalized. Size is (C, T, H, W)
        """
        return normalize(clip, self.mean, self.std, self.inplace)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(mean={self.mean}, std={self.std}, inplace={self.inplace})"


class ToTensorVideo:
    """Convert uint8 (T,C,H,W) tensor to float32 by dividing by 255."""

    def __init__(self):
        pass

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor, dtype=torch.uint8): Size is (T, C, H, W)
        Return:
            clip (torch.tensor, dtype=torch.float): Size is (T, C, H, W)
        """
        return to_tensor(clip)

    def __repr__(self) -> str:
        return self.__class__.__name__


class RandomHorizontalFlipVideo:
    """Flip the video clip horizontally with probability p."""

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Size is (T, C, H, W)
        Return:
            clip (torch.tensor): Size is (T, C, H, W)
        """
        if random.random() < self.p:
            clip = hflip(clip)
        return clip

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(p={self.p})"


class TemporalRandomCrop(object):
    """Randomly crop a temporal window of given length from frame indices."""

    def __init__(self, size):
        self.size = size

    def __call__(self, total_frames):
        rand_end = max(0, total_frames - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, total_frames)
        return begin_index, end_index


if __name__ == "__main__":
    import os

    import numpy as np
    import torchvision.io as io
    from torchvision import transforms
    from torchvision.utils import save_image

    vframes, aframes, info = io.read_video(filename="./v_Archery_g01_c03.avi", pts_unit="sec", output_format="TCHW")

    trans = transforms.Compose(
        [
            ToTensorVideo(),
            RandomHorizontalFlipVideo(),
            UCFCenterCropVideo(512),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        ]
    )

    target_video_len = 32
    frame_interval = 1
    total_frames = len(vframes)
    print(total_frames)

    temporal_sample = TemporalRandomCrop(target_video_len * frame_interval)

    # Sampling video frames
    start_frame_ind, end_frame_ind = temporal_sample(total_frames)
    assert end_frame_ind - start_frame_ind >= target_video_len
    frame_indice = np.linspace(start_frame_ind, end_frame_ind - 1, target_video_len, dtype=int)
    print(frame_indice)

    select_vframes = vframes[frame_indice]
    print(select_vframes.shape)
    print(select_vframes.dtype)

    select_vframes_trans = trans(select_vframes)
    print(select_vframes_trans.shape)
    print(select_vframes_trans.dtype)

    select_vframes_trans_int = ((select_vframes_trans * 0.5 + 0.5) * 255).to(dtype=torch.uint8)
    print(select_vframes_trans_int.dtype)
    print(select_vframes_trans_int.permute(0, 2, 3, 1).shape)

    io.write_video("./test.avi", select_vframes_trans_int.permute(0, 2, 3, 1), fps=8)

    for i in range(target_video_len):
        save_image(
            select_vframes_trans[i], os.path.join("./test000", "%04d.png" % i), normalize=True, value_range=(-1, 1)
        )
