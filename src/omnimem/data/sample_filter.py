import json
import random
from io import BytesIO

import numpy as np
from PIL import ImageFile

from .read_video import read_video
from .webdataset_utils import log_and_continue
from .utils import DataBucket, logging


ImageFile.LOAD_TRUNCATED_IMAGES = True


class SampleFilter:
    def __init__(
            self,
            resample_fps=1,
            match_aspect_ratio=True,
            frames_count_option=None,
            frame_aspect_ratio_option=None,
            aspect_ratio_size_option=None,
            min_aesthetic_score=None,
            dp_seed=0,
            possible_batch_size=None
    ):

        assert frames_count_option is not None, 'Please specify bucket information'
        assert frame_aspect_ratio_option is not None, 'Please specify bucket information'
        assert aspect_ratio_size_option is not None, 'Please specify bucket information'

        self.resample_fps = resample_fps
        self.match_aspect_ratio = match_aspect_ratio
        self.frames_count_option = frames_count_option
        self.frame_aspect_ratio_option = frame_aspect_ratio_option
        self.aspect_ratio_size_option = aspect_ratio_size_option
        self.min_aesthetic_score = min_aesthetic_score
        self.rng = random.Random(dp_seed)
        self.possible_batch_size = possible_batch_size

    @staticmethod
    def check_sample(sample):
        if 'image' not in sample and 'video' not in sample:
            return False

        if "json" not in sample and "text" not in sample and 'summary_text' not in sample:
            return False

        return True

    def __call__(self, sample):
        try:
            if not self.check_sample(sample):
                return False

            if 'text' in sample:
                sample['text'] = sample['text'].decode('utf-8')

            if 'summary_text' in sample:
                json_dict = json.loads(sample['summary_text'])
                sample['text'] = json_dict['text'].strip() if 'text' in json_dict else json_dict['txt']

            if 'video' in sample:

                info = json.loads(sample['json'])
                sample['text'] = info['caption'].strip()

                frames_counts = int(info['frame'])

                if frames_counts == 0:
                    return False

                info_fps = float(info['fps'])

                stride_resample = 1
                if self.resample_fps is not None and self.resample_fps > 0:
                    stride_resample = max(1, int(np.round(info_fps / self.resample_fps)))

                """ >>> choose num_frames first >>> """
                num_frames_candidate = [n for n in self.frames_count_option if n * stride_resample <= frames_counts]
                if len(num_frames_candidate) == 0:
                    logging.debug(
                        f"video too short - {stride_resample}:{min(self.frames_count_option)}/{frames_counts}")
                    return False
                if self.possible_batch_size is not None:
                    frame_weights = [sum(self.possible_batch_size[nf].values()) for nf in num_frames_candidate]
                else:
                    frame_weights = None
                num_frames = self.rng.choices(num_frames_candidate, weights=frame_weights, k=1)[0]

                sample['target_frames'] = num_frames

                ratio = float(info['height']) / float(info['width'])
                idx = np.abs(self.frame_aspect_ratio_option[num_frames] - ratio).argmin()
                ar = self.frame_aspect_ratio_option[num_frames][idx]
                error = abs(ar - ratio) if ratio > 1 else abs(1. / ar - 1. / ratio)

                if frames_counts < num_frames * stride_resample:
                    logging.debug(f"Video too short - {frames_counts} < {num_frames}x{stride_resample} ")
                    return False
                """ <<< choose num_frames first <<< """
                resolution_candidate = self.aspect_ratio_size_option[(num_frames, ar)]
                if self.possible_batch_size is not None:
                    resolution_weights = [self.possible_batch_size[num_frames][res] for res in resolution_candidate]
                else:
                    resolution_weights = None
                target_h, target_w = self.rng.choices(resolution_candidate, k=1, weights=resolution_weights)[0]

                if error > 0.3 and self.match_aspect_ratio:
                    logging.debug(
                        f"video aspect ratio not match - "
                        f"{info['height']}x{info['width']} -> {target_h}x{target_w}. skipping")
                    return False

                if int(info['height']) < target_h or int(info['width']) < target_w:
                    logging.debug(
                        f"Video size too small - "
                        f"{info['height']}x{info['width']}:{ratio} -> {target_h}x{target_w}:{ar}."
                        f" might cause some artifacts")

                # sample frame indices from clip
                total_span = (num_frames - 1) * stride_resample + 1
                start_frame_ind = self.rng.randint(0, max(0, frames_counts - total_span))
                frame_indice = start_frame_ind + np.arange(num_frames, dtype=int) * stride_resample
                if max(frame_indice) > frames_counts:
                    logging.debug(f"frame_indice > frames_counts - {frame_indice} > {frames_counts}")
                    return False

                # decode frames from video bytes
                with BytesIO(sample.pop('video')) as stream:
                    vframes, vinfo = read_video(stream, backend='decord', frame_index=frame_indice)
                if 'video_fps' not in vinfo and ('framerate' in info or 'fps' in info):
                    if 'framerate' in info:
                        vinfo['video_fps'] = float(info['framerate'])
                    else:
                        vinfo['video_fps'] = float(info['fps'])

                vinfo['video_fps'] = float(vinfo['video_fps'] / stride_resample)

                assert vframes.shape[1] == 3, 'Expect video shape to be (T, 3, H, W) but got {}'.format(vframes.shape)

                sample['video'] = vframes
                sample['vinfo'] = vinfo
            elif "image" in sample:
                raise NotImplementedError("image loading not implemented")
            else:
                raise NotImplementedError("loading func is not implemented")

            sample.update(dict(height=target_h, width=target_w, ar=ar))

            if 'text' not in sample and not isinstance(sample['text'], str):
                return False
            return True

        except Exception as exn:
            log_and_continue(exn)
            return False
