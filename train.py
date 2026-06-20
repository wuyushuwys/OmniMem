from omnimem.utils.numa_bind import bind_to_gpu_numa
bind_to_gpu_numa()

import argparse
import wandb
from omegaconf import OmegaConf
import torch.distributed as dist
from omnimem.trainer import TRAINER_CLS

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, default="config/config.yaml", help="path to config file")
    parser.add_argument("--name", type=str, required=True, help="Name of the experiment")
    parser.add_argument("--task", type=str, required=True, help="Which task to run")
    parser.add_argument("--evaluate", action="store_true", help="Evaluation only")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.name = args.name
    config.task = args.task

    if args.task in TRAINER_CLS:
        trainer = TRAINER_CLS[args.task](config=config)
    else:
        raise NotImplementedError(f"{args.task=} not implemented, expected {TRAINER_CLS.keys()}")

    try:
        trainer.logger.info(f"Start {trainer.__class__.__name__}")
        if args.evaluate:
            trainer.evaluate()
        else:
            trainer.train()
    except Exception as e:
        trainer.logger.error(e, exc_info=True)
        raise
    else:
        trainer.logger.info(f'End {args.task=}')
    finally:
        if wandb.run is not None:
            wandb.finish()
        if dist.is_initialized():
            dist.destroy_process_group()
