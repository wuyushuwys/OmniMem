import math
import os
import random
from collections import defaultdict
from collections.abc import MutableSequence
from pathlib import Path
from typing import Union, List
from urllib.parse import urlparse
import numpy as np
import torch
import webdataset as wds

import boto3
from torch.utils.data.distributed import DistributedSampler

from omnimem.utils.logging_tool import get_logger
from omnimem.data.webdataset_utils import (
    ResampledShards,
    SharedEpoch,
    log_and_continue
)

logger = get_logger()

class BaseDataset:
    sampler: DistributedSampler = None
    shared_epoch: SharedEpoch = None

    def __init__(self, *args, **kwargs):
        # each worker is iterating over this
        self._train_dataset = None
        self._train_dataloader = None

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def train_dataloader(self):
        return self._train_dataloader

    @property
    def dataloader(self):
        return self._train_dataloader

    def set_epoch(self, epoch):
        if self.shared_epoch is not None:
            self.shared_epoch.set_value(epoch)
        if self.sampler is not None and isinstance(self.sampler, DistributedSampler):
            self.sampler.set_epoch(epoch)


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


def collate_fn(examples, handler=log_and_continue):
    sample = defaultdict(list)
    for example in examples:
        for k, v in example.items():
            sample[k].append(v)

    for k, b in sample.items():
        if k != 't5_embed':
            try:
                if isinstance(b[0], (int, float)):
                    b = np.array(b)
                elif isinstance(b[0], torch.Tensor):
                    b = torch.stack(b)
                elif isinstance(b[0], np.ndarray):
                    b = np.stack(b)
            except Exception as e:
                logger.error(e, sample['__key__'], sample['__url__'])
                handler(e)

        if isinstance(b, list) and k not in ('__key__', '__url__', 't5_embed'):
            try:
                b = torch.stack([torch.as_tensor(x) for x in b])
            except Exception:
                pass

        sample[k] = b

    return sample


class OdeDataloader(BaseDataset):
    def __init__(
            self,
            train_shards_path_or_url: Union[str, List[str]],
            num_train_examples: int,
            per_gpu_batch_size: int,
            global_batch_size: int,
            num_workers: int,
            item_config=dict(ode="ode.pyd", timesteps="timesteps.pyd", t5_embed="t5_embed.pyd"),
            shuffle_buffer_size: int = 64,
            shuffle_init_size: int = 8,
            pin_memory: bool = False,
            persistent_workers: bool = False,
            prefetch_factor: int = 2,
            epoch: int = 1,
            group=None,
            **kwargs
    ):
        super().__init__()
        if not isinstance(train_shards_path_or_url, MutableSequence):
            train_shards_path_or_url = [train_shards_path_or_url]
        shards = []
        for data_folder in train_shards_path_or_url:
            if data_folder.startswith("s3://"):
                data_folder = data_folder.rstrip("/")
                parent_path = Path(data_folder).parts[
                    -1]
                parsed_url = urlparse(data_folder, allow_fragments=False)
                bucket_name = parsed_url.netloc
                s3_key = parsed_url.path.lstrip(
                    "/") + "?" + parsed_url.query if parsed_url.query else parsed_url.path.lstrip("/")
                s3 = boto3.resource("s3")
                my_bucket = s3.Bucket(bucket_name)
                for object_summary in my_bucket.objects.filter(Prefix=s3_key):
                    if object_summary.key.endswith(".tar") and Path(object_summary.key).parts[-2] == parent_path:
                        shards.append(f"pipe:aws s3 cp --quiet --cli-read-timeout 0 --cli-connect-timeout 60 s3://{bucket_name}/{object_summary.key} -")
            else:
                shards += [
                    os.path.join(data_folder, x) for x in os.listdir(data_folder) if x.endswith(".tar")
                ]

        """
        "__key__"
        "ode.pyd"
        "timesteps.pyd"
        "t5_embed.pyd"
        """

        # build pipeline and dataloader
        shard_epoch = SharedEpoch(epoch=epoch)
        pipeline = [
            ResampledShards(
                shards,
                deterministic=False,
                epoch=shard_epoch,
            ),
            detshuffle2(
                bufsize=32,
                initial=4,
                seed=42,
                epoch=shard_epoch,
            ),
            wds.shardlists.split_by_node,
            wds.shardlists.split_by_worker,
            wds.tarfile_to_samples(handler=wds.handlers.ignore_and_continue),
            wds.decode(handler=log_and_continue),
            wds.rename(
                **item_config,
                handler=log_and_continue,
            ),
            wds.shuffle(
                bufsize=shuffle_buffer_size,
                initial=shuffle_init_size,
            ),
        ]

        effective_workers = max(1, num_workers)

        num_worker_batches = math.ceil(
            num_train_examples / (global_batch_size * effective_workers)
        )  # per dataloader worker
        num_batches = num_worker_batches * effective_workers
        num_samples = num_batches * global_batch_size

        self._train_dataset = wds.DataPipeline(*pipeline)
        self._train_dataloader = (
            wds.WebLoader(
                self._train_dataset,
                batch_size=None,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory,
                prefetch_factor=prefetch_factor,
                persistent_workers=persistent_workers,
            )
            .with_epoch(num_batches)
            .batched(per_gpu_batch_size, collation_fn=collate_fn)
        )
        self._train_dataloader.num_batches = num_batches
        self._train_dataloader.num_samples = num_samples
