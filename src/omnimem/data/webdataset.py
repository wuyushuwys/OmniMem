import time
import copy
import random
import os
import tarfile
from itertools import islice
from typing import Iterable, Any, Callable, Optional, List
from typing import Dict, Iterator, Union
from collections import defaultdict
from pathlib import Path
from urllib.parse import urlparse

import boto3
import numpy as np

from PIL import ImageFile

import torch
import torch.distributed as dist
import torch.utils.data
from torch.utils.data import IterableDataset

import webdataset as wds
from webdataset.handlers import reraise_exception
from webdataset.filters import pipelinefilter, getfirst
from webdataset.tariterators import url_opener, valid_sample, meta_suffix, meta_prefix
from webdataset import gopen

from omnimem.data.webdataset_utils import DataInfo, SharedEpoch, log_and_continue
from omnimem.data.utils import DataBucket

from .utils import logging
from .sample_filter import SampleFilter
from .sample_transform import SampleTransform

ImageFile.LOAD_TRUNCATED_IMAGES = True

_SHARD_SHUFFLE_SIZE = 32
_SHARD_SHUFFLE_INITIAL = 16
_SAMPLE_SHUFFLE_SIZE = 50  # some video can be 40mb
_SAMPLE_SHUFFLE_INITIAL = 25

image_metadata_extension = ['.txt']
video_metadata_extension = [".json"]
video_extensions = [".mp4", ".webp", ".webm", ".mov", ".mkv", ".avi"]
image_extensions = [".png", ".jpg", ".jpeg", '.webp', '.data']  # ".gif"


def tar_file_iterator(
        fileobj: tarfile.TarFile,
        handler: Callable[[Exception], bool],
        select_files: Optional[Callable[[str], bool]] = None,
        rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    """
    for tar with raw videos
    """
    stream = tarfile.open(fileobj=fileobj, mode="r|*")
    for tarinfo in stream:
        fname = tarinfo.name
        try:
            if not tarinfo.isreg():
                continue
            if fname is None:
                continue
            if (
                    "/" not in fname
                    and fname.startswith(meta_prefix)
                    and fname.endswith(meta_suffix)
            ):
                continue
            if fname.count('.') > 1 and '.json' in fname:
                fname = fname.replace('.json', '')
            fname = Path(fname)
            allowed_suffix = ['.json', '.txt', '.gzip'] + video_extensions + video_metadata_extension + image_extensions
            if fname.suffix.lower() not in allowed_suffix:
                continue

            parts = fname.parts[-2:]
            fname = os.path.join(*parts)
            data = stream.extractfile(tarinfo).read()
            result = dict(fname=fname, data=data)
            yield result
            stream.members = []
        except Exception as exn:
            if hasattr(exn, "args") and len(exn.args) > 0:
                exn.args = (str(exn.args[0]) + " @ " + str(fileobj),) + exn.args[1:]
            if handler(exn):
                continue
            else:
                break
    del stream


# pair text / text embedding -> image / video
def tar_file_expander(
        data: Iterable[Dict[str, Any]],
        handler: Callable[[Exception], bool],
        select_files: Optional[Callable[[str], bool]] = None,
        rename_files: Optional[Callable[[str], str]] = None,
) -> Iterator[Dict[str, Any]]:
    for source in data:
        url = source["url"]
        try:
            assert isinstance(source, dict)
            assert "stream" in source
            for sample in tar_file_iterator(
                    source["stream"],
                    handler=handler,
                    select_files=select_files,
                    rename_files=rename_files,
            ):
                assert isinstance(sample, dict) and "data" in sample and "fname" in sample
                sample["__url__"] = url
                yield sample
        except Exception as exn:
            exn.args = exn.args + (source.get("stream"), source.get("url"))
            if handler(exn):
                continue
            else:
                raise exn


def group_by_keys_nothrow(data, lcase=True, suffixes=None, handler=None):
    current_sample = None
    for filesample in data:
        try:
            assert isinstance(filesample, dict)
            fname, value = filesample["fname"], filesample["data"]

            last_dot_index = fname.rfind(".")
            if last_dot_index != -1:
                prefix = fname[:last_dot_index]
                suffix = fname[last_dot_index + 1:]
            else:
                continue

            if prefix is None or len(prefix) == 0:
                continue

            if lcase:
                suffix = suffix.lower()

            if current_sample is None or prefix != current_sample["__key__"] or suffix in current_sample:
                if valid_sample(current_sample):
                    yield current_sample
                current_sample = dict(__key__=prefix, __url__=filesample["__url__"])
            if suffixes is None or suffix in suffixes:
                current_sample[suffix] = value
        except Exception as exn:
            if handler(exn):
                continue
            else:
                raise exn
    if valid_sample(current_sample):
        yield current_sample


def tarfile_to_samples_nothrow(src, handler=log_and_continue):
    # open tarfiles; each sample['stream'] is a tarfile
    streams = url_opener(src, handler=handler)
    files = tar_file_expander(streams, handler=handler)

    samples = group_by_keys_nothrow(files, handler=handler)
    return samples


class detshuffle2(wds.PipelineStage):
    def __init__(
            self,
            bufsize=1000,
            initial=100,
            seed=0,
            epoch=-1,
    ):
        self.bufsize = bufsize
        self.initial = initial
        self.seed = seed
        self.epoch: SharedEpoch = epoch

    def run(self, src):
        epoch = self.epoch.get_value()
        seed = self.seed + epoch
        rng = random.Random()
        rng.seed(seed)
        return wds.filters._shuffle(src, self.bufsize, self.initial, rng)


class SimplePairShardList(IterableDataset):
    """An iterable dataset yielding a list of urls."""

    def __init__(self, paired_urls, seed=None, weights=None):
        super().__init__()
        self.urls = paired_urls
        assert isinstance(self.urls[0], str)
        self.seed = seed
        self.weights = weights
        if weights is not None:
            assert len(self.urls) == len(self.weights), f"{len(self.urls)} != {len(self.weights)}"
        self.rng = random.Random(self.seed)

    def __len__(self):
        return len(self.urls)

    def __iter__(self):
        urls = copy.deepcopy(self.urls)
        if self.weights is not None:
            for _ in range(len(urls)):
                url = self.rng.choices(urls, weights=self.weights, k=1)[0]
                yield dict(url=url)
        else:
            if self.seed is not None:
                self.rng.shuffle(urls)
            for url in urls:
                yield dict(url=url)


def url_opener(
        data: Iterable[Dict[str, Any]],
        handler: Callable[[Exception], bool],
        **kw: Dict[str, Any],
):
    for sample in data:
        assert isinstance(sample, dict), sample
        assert "url" in sample, sample
        url = sample["url"]
        stream_ready = False
        max_retires = 1
        handle_failed = False

        while max_retires > 0 and not stream_ready:
            try:
                if not stream_ready:
                    stream = gopen(url, **kw)
                    sample.update(stream=stream)
                    stream_ready = True
                yield sample
            except Exception as exn:
                max_retires -= 1
                logging.warning(f'data download failed: retrying {1 - max_retires}th time')
                if max_retires == 0:
                    exn.args = exn.args + (url,)
                    handle_failed = not handler(exn, "url_opener")
        if handle_failed:
            break


def collate_fn(examples):
    sample = defaultdict(list)
    for example in examples:
        for k, v in example.items():
            sample[k].append(v)

    for k, b in sample.items():
        if isinstance(b[0], (int, float)):
            b = np.array(b)
        elif isinstance(b[0], torch.Tensor):
            b = torch.stack(b)
        elif isinstance(b[0], np.ndarray):
            b = np.stack(b)
        sample[k] = b

    return sample


def _rename(data, handler=reraise_exception, **kw):
    for sample in data:
        try:
            def listify(v):
                return v.split(";") if isinstance(v, str) else v

            to_be_replaced = {x for v in kw.values() for x in listify(v)}
            result = {k: v for k, v in sample.items() if k not in to_be_replaced}
            renamed = {
                k: getfirst(sample, v, default=None, missing_is_error=False)
                for k, v in kw.items()
            }
            renamed = {k: v for k, v in renamed.items() if v is not None}
            result.update(renamed)
            yield result
        except Exception as exn:
            if handler(exn):
                continue
            else:
                break


def _batched_bucket(
        data,
        buckets: DataBucket,
        collation_fn=collate_fn,
        partial=False,
):
    for sample in data:
        num_frames = int(sample['num_frames'])
        if num_frames not in buckets:
            logging.warning(f'num_frames not in buckets: {num_frames}')
            continue
        resolution = (int(sample['height']), int(sample['width']))
        if resolution not in buckets(num_frames):
            logging.warning(f'resolution not in buckets: {resolution} -- [{num_frames}]: {buckets(num_frames).keys()}')
            continue
        batch = buckets(num_frames)[resolution]
        batch_size = buckets.batch_size[num_frames][resolution]
        batch.append(sample)
        if len(batch) >= batch_size:
            if collation_fn is not None:
                batch = collation_fn(batch)
            yield batch
            buckets(num_frames)[resolution] = []

    if partial:
        # don't support
        raise NotImplementedError(f'partial bucket not implemented not got {partial}')


def split_by_node(src, group=None):
    rank = 0
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        rank = torch.distributed.get_rank(group=group)
        world_size = torch.distributed.get_world_size(group=group)
    if world_size > 1:
        yield from islice(src, rank, None, world_size)
    else:
        yield from src


def split_by_worker(src):
    worker = 0
    num_workers = 1
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        worker = worker_info.id
        num_workers = worker_info.num_workers
    if num_workers > 1:
        yield from islice(src, worker, None, num_workers)
    else:
        yield from src


def pick(buf, rng):
    k = rng.randint(0, len(buf) - 1)
    sample = buf[k]
    buf[k] = buf[-1]
    buf.pop()
    return sample


def _shuffle(data, bufsize=1000, initial=100, seed=22, handler=None):
    rng = random.Random()
    rng.seed(seed)
    initial = min(initial, bufsize)
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) < bufsize:
            try:
                buf.append(next(data))
            except StopIteration:
                pass
        if len(buf) >= initial:
            yield pick(buf, rng)
    while len(buf) > 0:
        yield pick(buf, rng)


def prepare_wds_dataloader(
        data_bucket: Union[DataBucket, Dict],
        paired_shards: List[str] = None,
        sample_shuffle_initial: Optional[int] = _SAMPLE_SHUFFLE_INITIAL,
        sample_shuffle_size: Optional[int] = _SAMPLE_SHUFFLE_SIZE,
        shard_shuffle_initial: Optional[int] = _SHARD_SHUFFLE_INITIAL,
        shard_shuffle_size: Optional[int] = _SHARD_SHUFFLE_SIZE,
        match_aspect_ratio: bool = False,
        resample_fps: float = 24,
        prefetch_factor: Optional[int] = 2,
        epoch: Optional[int] = 0,
        seed: Optional[int] = None,
        group: Optional[int] = None,
        num_workers: Optional[int] = 8,
        persistent_workers: Optional[bool] = True,
        pin_memory: Optional[bool] = False,
        resize_method: Optional[str] = 'resize_crop_tv',
        min_aesthetic_score: Optional[float] = None,
        null_condition_prob: Optional[float] = 0.,
        data_urls_weight: Optional[List[float]] = None,
):
    """Build a WebDataset-based DataLoader for video/image training.

    Returns:
        DataInfo: dataloader and shared epoch.
    """
    if not isinstance(data_bucket, DataBucket):
        data_bucket = DataBucket(data_bucket)

    frames_options = data_bucket.frames_options
    aspect_ratio_size = data_bucket.aspect_ratio_size
    frame_aspect_ratio = data_bucket.frame_aspect_ratio

    shared_epoch = SharedEpoch(epoch=epoch)

    seed = time.time_ns() % 2 ** 32 if seed is None else seed
    tensor_seed = torch.tensor(seed, dtype=torch.int64, device='cuda')
    if dist.is_initialized() and dist.get_world_size() > 1:
        dist.barrier()
        dist.broadcast(tensor_seed, src=0)
    seed = int(tensor_seed.cpu().numpy())
    logging.info(f'data shuffle seed: {seed}')
    np.random.seed(seed)
    if data_urls_weight is None:
        np.random.shuffle(paired_shards)

    pipeline = []
    rank = torch.distributed.get_rank(group=group) if dist.is_initialized() else 0
    dp_rank_seed = seed + 22 * rank

    pipeline.extend([
        SimplePairShardList(paired_shards, seed=seed, weights=data_urls_weight),
        detshuffle2(
            bufsize=shard_shuffle_size,
            initial=shard_shuffle_initial,
            seed=seed,
            epoch=shared_epoch,
        ),
        pipelinefilter(wds.shardlists.split_by_node)(group=group),
        wds.shardlists.split_by_worker,
        pipelinefilter(tarfile_to_samples_nothrow)(),
        pipelinefilter(_shuffle)(bufsize=sample_shuffle_size,
                                 initial=sample_shuffle_initial, seed=dp_rank_seed),
    ])

    pipeline.extend([
        pipelinefilter(_rename)(video="mp4;webp;webm;mov;mkv;avi", image="jpg;png;jpeg;webp;data", text="txt"),
        wds.select(
            SampleFilter(
                resample_fps=resample_fps,
                frames_count_option=frames_options,
                aspect_ratio_size_option=aspect_ratio_size,
                frame_aspect_ratio_option=frame_aspect_ratio,
                match_aspect_ratio=match_aspect_ratio,
                min_aesthetic_score=min_aesthetic_score,
                dp_seed=dp_rank_seed,
                possible_batch_size=data_bucket.batch_size,
            )
        ),
        wds.map(
            SampleTransform(
                resize_method=resize_method,
                null_condition_prob=null_condition_prob,
                dp_seed=dp_rank_seed,
            )
        ),
        pipelinefilter(_batched_bucket)(data_bucket, partial=False)

    ])

    dataset = wds.DataPipeline(*pipeline)

    dataloader = wds.WebLoader(
        dataset,
        batch_size=None,
        shuffle=False,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    return DataInfo(dataloader=dataloader, shared_epoch=shared_epoch)


def parse_shards(data_path, ratio=None):
    if not isinstance(data_path, list):
        data_path = [data_path]
    shards = []
    for data_folder in data_path:
        if data_folder.startswith("s3://"):
            data_folder = data_folder.rstrip("/")
            parent_path = Path(data_folder).parts[-1]
            parsed_url = urlparse(data_folder, allow_fragments=False)
            bucket_name = parsed_url.netloc
            s3_key = parsed_url.path.lstrip(
                "/") + "?" + parsed_url.query if parsed_url.query else parsed_url.path.lstrip("/")
            s3 = boto3.resource("s3")
            my_bucket = s3.Bucket(bucket_name)
            logging.info(f"{my_bucket} -- {s3_key}")
            for object_summary in my_bucket.objects.filter(Prefix=s3_key):
                if object_summary.key.endswith(".tar") and Path(object_summary.key).parts[-2] == parent_path:
                    shards.append(f"pipe:aws s3 cp --quiet --cli-read-timeout 0 --cli-connect-timeout 60 s3://{bucket_name}/{object_summary.key} -")
        else:
            shards += [
                os.path.join(data_folder, x) for x in os.listdir(data_folder) if x.endswith(".tar")
            ]
    if ratio is not None:
        rng = random.Random(dist.get_rank() + time.time_ns() if dist.is_initialized() else time.time_ns())
        if ratio > 1:
            shards = rng.choices(shards, k=int(len(shards) * ratio))
        else:
            shards = rng.sample(shards, k=int(len(shards) * ratio))
    return shards


if __name__ == '__main__':
    pass
