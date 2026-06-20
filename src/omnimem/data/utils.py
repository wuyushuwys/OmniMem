import re

from typing import Dict
from functools import lru_cache
from collections import defaultdict

import ftfy
import urllib.parse as ul
import html
import numpy as np
from bs4 import BeautifulSoup
from PIL import Image

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.v2 as transforms_v2
from torchvision.transforms.v2.functional import InterpolationMode

from omnimem.utils.logging_tool import get_logger
from omnimem.data import video_transforms

IMAGE_FPS = 120

VID_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv")

logging = get_logger()


regex = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"  # domain...
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


bad_punct_regex = re.compile(
    r"["
    + "#®•©™&@·º½¾¿¡§~"
    + r"\)"
    + r"\("
    + r"\]"
    + r"\["
    + r"\}"
    + r"\{"
    + r"\|"
    + "\\"
    + r"\/"
    + r"\*"
    + r"]{1,}"
)


def clean_caption(caption):
    caption = str(caption)
    caption = ul.unquote_plus(caption)
    caption = caption.strip().lower()
    caption = re.sub("<person>", "person", caption)
    caption = re.sub(
        r"\b((?:https?:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
        # noqa
        "",
        caption,
    )
    caption = re.sub(
        r"\b((?:www:(?:\/{1,3}|[a-zA-Z0-9%])|[a-zA-Z0-9.\-]+[.](?:com|co|ru|net|org|edu|gov|it)[\w/-]*\b\/?(?!@)))",
        # noqa
        "",
        caption,
    )
    caption = BeautifulSoup(caption, features="html.parser").text
    caption = re.sub(r"@[\w\d]+\b", "", caption)
    # Remove CJK character blocks
    caption = re.sub(r"[\u31c0-\u31ef]+", "", caption)
    caption = re.sub(r"[\u31f0-\u31ff]+", "", caption)
    caption = re.sub(r"[\u3200-\u32ff]+", "", caption)
    caption = re.sub(r"[\u3300-\u33ff]+", "", caption)
    caption = re.sub(r"[\u3400-\u4dbf]+", "", caption)
    caption = re.sub(r"[\u4dc0-\u4dff]+", "", caption)
    caption = re.sub(r"[\u4e00-\u9fff]+", "", caption)

    # все виды тире / all types of dash --> "-"
    caption = re.sub(
        r"[\u002D\u058A\u05BE\u1400\u1806\u2010-\u2015\u2E17\u2E1A\u2E3A\u2E3B\u2E40\u301C\u3030\u30A0\uFE31\uFE32\uFE58\uFE63\uFF0D]+",
        # noqa
        "-",
        caption,
    )

    caption = re.sub(r"[`´«»“”¨]", '"', caption)
    caption = re.sub(r"[‘’]", "'", caption)
    caption = re.sub(r"&quot;?", "", caption)
    caption = re.sub(r"&amp", "", caption)
    caption = re.sub(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}", " ", caption)
    caption = re.sub(r"\d:\d\d\s+$", "", caption)
    caption = re.sub(r"\\n", " ", caption)
    caption = re.sub(r"#\d{1,3}\b", "", caption)
    caption = re.sub(r"#\d{5,}\b", "", caption)
    caption = re.sub(r"\b\d{6,}\b", "", caption)
    # filenames:
    caption = re.sub(
        r"[\S]+\.(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)", "", caption
    )

    #
    caption = re.sub(r"[\"\']{2,}", r'"', caption)  # """AUSVERKAUFT"""
    caption = re.sub(r"[\.]{2,}", r" ", caption)  # """AUSVERKAUFT"""

    caption = re.sub(
        bad_punct_regex, r" ", caption
    )  # ***AUSVERKAUFT***, #AUSVERKAUFT
    caption = re.sub(r"\s+\.\s+", r" ", caption)  # " . "

    regex2 = re.compile(r"(?:\-|\_)")
    if len(re.findall(regex2, caption)) > 3:
        caption = re.sub(regex2, " ", caption)

    caption = ftfy.fix_text(caption)
    caption = html.unescape(html.unescape(caption))

    caption = re.sub(r"\b[a-zA-Z]{1,3}\d{3,15}\b", "", caption)  # jc6640
    caption = re.sub(r"\b[a-zA-Z]+\d+[a-zA-Z]+\b", "", caption)  # jc6640vc
    caption = re.sub(r"\b\d+[a-zA-Z]+\d+\b", "", caption)  # 6640vc231

    caption = re.sub(r"(worldwide\s+)?(free\s+)?shipping", "", caption)
    caption = re.sub(r"(free\s)?download(\sfree)?", "", caption)
    caption = re.sub(r"\bclick\b\s(?:for|on)\s\w+", "", caption)
    caption = re.sub(
        r"\b(?:png|jpg|jpeg|bmp|webp|eps|pdf|apk|mp4)(\simage[s]?)?", "", caption
    )
    caption = re.sub(r"\bpage\s+\d+\b", "", caption)

    caption = re.sub(
        r"\b\d*[a-zA-Z]+\d+[a-zA-Z]+\d+[a-zA-Z\d]*\b", r" ", caption
    )  # j2d1a2a...

    caption = re.sub(r"\b\d+\.?\d*[xх×]\d+\.?\d*\b", "", caption)

    caption = re.sub(r"\b\s+\:\s+", r": ", caption)
    caption = re.sub(r"(\D[,\./])\b", r"\1 ", caption)
    caption = re.sub(r"\s+", " ", caption)

    caption.strip()

    caption = re.sub(r"^[\"\']([\w\W]+)[\"\']$", r"\1", caption)
    caption = re.sub(r"^[\'\_,\-\:;]", r"", caption)
    caption = re.sub(r"[\'\_,\-\:\-\+]$", r"", caption)
    caption = re.sub(r"^\.\S+$", "", caption)

    return caption.strip()


class DataBucket:
    """
            example:

            BUCKETS = {
                "16": {
                    "768x432": 4,
                    "640x480": 4,
                    "512x512": 4,
                    "432x768": 4,
                    "480x640": 4,
                },
                "8": {
                    "768x432": 6,
                    "640x480": 6,
                    "512x512": 6,
                    "432x768": 6,
                    "480x640": 6,

                },
                "1": {
                    "768x432": 32,
                    "640x480": 32,
                    "512x512": 32,
                    "432x768": 32,
                    "480x640": 32,
                },
            }

            """
    possible_aspect_ratio_candidate_greater_than_one = np.array([4 / 3, 16 / 9, 1])

    def __init__(self, bucket):
        self.bucket = {}
        self._batch_size = {}
        self._aspect_ratio_size = {}
        self._frame_aspect_ratio = {}
        for n_frames, f_bucket_info in bucket.items():
            self.bucket[int(n_frames)] = defaultdict()
            self._batch_size[int(n_frames)] = defaultdict()
            for res, bsz in f_bucket_info.items():
                h, w = res.split('x')
                h = int(h)
                w = int(w)
                self.bucket[int(n_frames)][(h, w)] = []
                self._batch_size[int(n_frames)][(h, w)] = bsz
                ar = self.approx_ar(h / w)
                if (int(n_frames), ar) not in self._aspect_ratio_size:
                    self._aspect_ratio_size[(int(n_frames), ar)] = [(h, w)]
                else:
                    if (h, w) not in self._aspect_ratio_size[(int(n_frames), ar)]:
                        self._aspect_ratio_size[(int(n_frames), ar)].append((h, w))
                if int(n_frames) not in self._frame_aspect_ratio:
                    self._frame_aspect_ratio[int(n_frames)] = [ar]
                else:
                    if ar not in self._frame_aspect_ratio[int(n_frames)]:
                        self._frame_aspect_ratio[int(n_frames)].append(ar)

    def approx_ar(self, ar):
        flipped = False
        if ar < 1.0:
            ar = 1.0 / ar
            flipped = True
        idx = np.abs(self.possible_aspect_ratio_candidate_greater_than_one - ar).argmin()
        approx_ar = self.possible_aspect_ratio_candidate_greater_than_one[idx]
        if flipped:
            approx_ar = float(1.0 / approx_ar)
        return approx_ar

    def __call__(self, key) -> Dict:
        return self.bucket.get(key)

    @property
    def frames_options(self):
        return [int(k) for k in self.bucket.keys() if int(k) > 1]

    @property
    def aspect_ratio_size(self):
        return self._aspect_ratio_size

    @property
    def aspect_ratio(self):
        return np.unique([ar for n_frames, ar in self.aspect_ratio_size.keys()])

    @property
    def frame_aspect_ratio(self):
        frame_aspect_ratio = {k: np.array(v) for k, v in self._frame_aspect_ratio.items()}
        return frame_aspect_ratio

    def __contains__(self, item):
        return item in self.bucket

    @property
    def batch_size(self):
        return self._batch_size

    def extend_batch_size(self, factor):
        for n_frames, info in self._batch_size.items():
            for res, bsz in info.items():
                self._batch_size[n_frames][res] *= 2


@lru_cache
def get_transforms_video(name="center", image_size=(256, 256)):
    if name is None:
        return None
    elif name == "center":
        assert image_size[0] == image_size[1], "image_size must be square for center crop"
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.UCFCenterCropVideo(image_size[0]),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop":
        transform_video = transforms.Compose(
            [
                video_transforms.ToTensorVideo(),  # TCHW
                video_transforms.ResizeCrop(image_size),
                transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "resize_crop_tv":

        transform_video = transforms.Compose(
            [
                transforms_v2.ToDtype(torch.uint8, scale=True),
                video_transforms.ResizeCrop(image_size, 'bicubic'),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    elif name == "random_crop":

        transform_video = transforms.Compose(
            [
                transforms_v2.ToDtype(torch.uint8, scale=True),
                video_transforms.RandomCropVideo(image_size),
                transforms_v2.ToDtype(torch.float32, scale=True),
                transforms_v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
            ]
        )
    else:
        raise NotImplementedError(f"Transform {name} not implemented")
    return transform_video


def resize_crop_to_fill(pil_image: Image.Image, image_size, mode=Image.LANCZOS):
    w, h = pil_image.size  # PIL is (W, H)
    th, tw = image_size
    rh, rw = th / h, tw / w
    if rh > rw:
        sh, sw = th, round(w * rh)
        image = pil_image.resize((sw, sh), mode)
        i = 0
        j = int(round((sw - tw) / 2.0))
    else:
        sh, sw = round(h * rw), tw
        image = pil_image.resize((sw, sh), mode)
        i = int(round((sh - th) / 2.0))
        j = 0

    assert i + th <= image.size[1] and j + tw <= image.size[0]
    return image.crop((j, i, j + tw, i + th))


