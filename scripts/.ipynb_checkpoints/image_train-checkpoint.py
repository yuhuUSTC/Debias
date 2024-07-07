"""
Train a diffusion model on images.
"""

import argparse
import os
import torch
from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop



def main():
    args = create_argparser().parse_args()
    torch.distributed.init_process_group(backend='nccl')
    #dist_util.setup_dist()
    logger.configure(dir=args.log_dir)

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    #model.load_state_dict(dist_util.load_state_dict(args.model_path1, map_location="cpu"))
    model.to(dist_util.dev())

    # model2, _ = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    # model2.load_state_dict(dist_util.load_state_dict(args.model_path2, map_location="cpu"))
    # model2.to(dist_util.dev())

    # model3, _ = create_model_and_diffusion(**args_to_dict(args, model_and_diffusion_defaults().keys()))
    # model3.load_state_dict(dist_util.load_state_dict(args.model_path3, map_location="cpu"))
    # model3.to(dist_util.dev())

    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(data_dir=args.data_dir, batch_size=args.batch_size, image_size=args.image_size, class_cond=args.class_cond,)

    logger.log("training...")
    TrainLoop(
        model=model,
        #model2=[model2,model3],
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="",
        log_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=1000,
        save_interval=50000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        predict_xstart=False,
        predict_v=False,
        num_classes=1000,
        model_path1="",       
        model_path2="",
        model_path3="",
        #local_rank=-1,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', default=-1, type=int, help='node rank for distributed training')
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
