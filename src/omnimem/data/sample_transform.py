import random
import einops

from omnimem.data.utils import get_transforms_video, clean_caption


class SampleTransform:
    def __init__(
            self,
            resize_method="resize_crop",
            null_condition_prob: float = 0.,
            dp_seed=0,
    ):
        self.rng = random.Random(dp_seed)
        self.resize_method = resize_method
        self.null_condition_prob = null_condition_prob

        self.cache_transform = {}

    def __call__(self, sample):
        assert 'text' in sample, sample.keys()

        if 'image' in sample:
            raise NotImplementedError("image not implemented")
        elif 'video' in sample:
            vinfo = sample.pop('vinfo')
            video_fps = vinfo["video_fps"] if "video_fps" in vinfo else 24.
            video = sample.pop('video')
            num_frames = sample.pop('target_frames')
            if (sample['height'], sample['width']) not in self.cache_transform:
                self.cache_transform[(sample['height'], sample['width'])] = get_transforms_video(
                    self.resize_method,
                    (sample['height'], sample['width']),
                )
            video = self.cache_transform[(sample['height'], sample['width'])](video)
            frame_count = num_frames
        else:
            raise NotImplementedError(f'sample {sample.keys()}')

        video = einops.rearrange(video, "t c h w -> c t h w")
        sample.update({
            "pixel_values": video,
            "num_frames": frame_count,
            "fps": video_fps,
            "text": clean_caption(sample["text"]) if self.rng.random() >= self.null_condition_prob else '',
        })
        return sample
