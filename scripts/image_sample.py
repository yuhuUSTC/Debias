"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""

import argparse
import os
import random
import torch
import numpy as np
import torch as th
import torch.distributed as dist
from einops import rearrange
from PIL import Image

from guided_diffusion import dist_util, logger
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    add_dict_to_argparser,
    args_to_dict,
)
from torchvision import utils

def seed_torch(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False


def main():
    seed_torch(seed=1000)

    args = create_argparser().parse_args()
    torch.distributed.init_process_group(backend='nccl')
    #sdist_util.setup_dist()
    if not os.path.exists(args.sample_dir):
        os.makedirs(args.sample_dir, exist_ok=True)
    logger.configure(dir=args.sample_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    model.load_state_dict(dist_util.load_state_dict(args.model_path, map_location="cpu"))
    model.to(dist_util.dev())
    if args.use_fp16:
        model.convert_to_fp16()
    model.eval()

    logger.log("sampling...")
    all_images = []
    all_labels = []
    count = 0
    while count * args.batch_size < args.num_samples:
        model_kwargs = {}
        if args.class_cond:
            classes = th.randint(low=0, high=args.num_classes, size=(args.batch_size,), device=dist_util.dev())
            model_kwargs["y"] = classes
        
        
        if args.sampler == "ddpm":
            sample_fn = diffusion.p_sample_loop
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        elif args.sampler == "ddim":
            sample_fn = diffusion.ddim_sample_loop
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                clip_denoised=args.clip_denoised,
                model_kwargs=model_kwargs,
            )
        elif args.sampler == "dpm_solver":
            sample_fn = diffusion.dpm_solver_uncond_loop
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                model_kwargs=model_kwargs,
                timestep_respacing=int(args.timestep_respacing),
                )
        if args.sampler == "unipc":
            sample_fn = diffusion.UniPC_uncond_loop
            sample = sample_fn(
                model,
                (args.batch_size, 3, args.image_size, args.image_size),
                model_kwargs=model_kwargs,
                timestep_respacing=int(args.timestep_respacing),
            )
        for i in range(args.batch_size):
            out_path = os.path.join(logger.get_dir(), f"{str(count * args.batch_size + i).zfill(5)}.png")
            utils.save_image(
                sample[i].unsqueeze(0),
                out_path,
                nrow=1,
                normalize=True,
                value_range=(-1, 1),
                #range=(-1, 1),
            )
        count += 1


    dist.barrier()
    logger.log("sampling complete")


def create_argparser():
    defaults = dict(
        clip_denoised=True,
        num_samples=50000,
        batch_size=32,
        use_ddim=False,
        model_path="",
        sample_dir="",
        num_classes=1,
        sampler="ddpm",
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()