# Unmasking Bias in Diffusion Model Training

<div align="center">

[![demo platform](https://img.shields.io/badge/Play%20with%20VAR%21-VAR%20demo%20platform-lightblue)](https://var.vision/demo)&nbsp;
[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2404.02905)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-FoundationVision/var-yellow)](https://huggingface.co/FoundationVision/var)&nbsp;
[![SOTA](https://img.shields.io/badge/State%20of%20the%20Art-Image%20Generation%20on%20ImageNet%20%28AR%29-32B1B4?logo=data%3Aimage%2Fsvg%2Bxml%3Bbase64%2CPHN2ZyB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIgeG1sbnM9Imh0dHA6Ly93d3cudzMub3JnLzIwMDAvc3ZnIiB4bWxuczp4bGluaz0iaHR0cDovL3d3dy53My5vcmcvMTk5OS94bGluayIgb3ZlcmZsb3c9ImhpZGRlbiI%2BPGRlZnM%2BPGNsaXBQYXRoIGlkPSJjbGlwMCI%2BPHJlY3QgeD0iLTEiIHk9Ii0xIiB3aWR0aD0iNjA2IiBoZWlnaHQ9IjYwNiIvPjwvY2xpcFBhdGg%2BPC9kZWZzPjxnIGNsaXAtcGF0aD0idXJsKCNjbGlwMCkiIHRyYW5zZm9ybT0idHJhbnNsYXRlKDEgMSkiPjxyZWN0IHg9IjUyOSIgeT0iNjYiIHdpZHRoPSI1NiIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIxOSIgeT0iNjYiIHdpZHRoPSI1NyIgaGVpZ2h0PSI0NzMiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIyNzQiIHk9IjE1MSIgd2lkdGg9IjU3IiBoZWlnaHQ9IjMwMiIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjEwNCIgeT0iMTUxIiB3aWR0aD0iNTciIGhlaWdodD0iMzAyIiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNDQ0IiB5PSIxNTEiIHdpZHRoPSI1NyIgaGVpZ2h0PSIzMDIiIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSIzNTkiIHk9IjE3MCIgd2lkdGg9IjU2IiBoZWlnaHQ9IjI2NCIgZmlsbD0iIzQ0RjJGNiIvPjxyZWN0IHg9IjE4OCIgeT0iMTcwIiB3aWR0aD0iNTciIGhlaWdodD0iMjY0IiBmaWxsPSIjNDRGMkY2Ii8%2BPHJlY3QgeD0iNzYiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjY2IiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI3NiIgeT0iNDgyIiB3aWR0aD0iNDciIGhlaWdodD0iNTciIGZpbGw9IiM0NEYyRjYiLz48cmVjdCB4PSI0ODIiIHk9IjQ4MiIgd2lkdGg9IjQ3IiBoZWlnaHQ9IjU3IiBmaWxsPSIjNDRGMkY2Ii8%2BPC9nPjwvc3ZnPg%3D%3D)](https://paperswithcode.com/sota/image-generation-on-imagenet-256x256?tag_filter=485&p=visual-autoregressive-modeling-scalable-image)


</div>
<p align="center" style="font-size: larger;">
  <a href="https://arxiv.org/abs/2404.02905">Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction</a>
</p>

<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/9850df90-20b1-4f29-8592-e3526d16d755" width=95%>
<p>

<br>

## What's New?

### 🔥 Introducing VAR: a new paradigm in autoregressive visual generation✨:

Visual Autoregressive Modeling (VAR) redefines the autoregressive learning on images as coarse-to-fine "next-scale prediction" or "next-resolution prediction", diverging from the standard raster-scan "next-token prediction".

<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/3e12655c-37dc-4528-b923-ec6c4cfef178" width=93%>
<p>

### 🔥 For the first time, GPT-style autoregressive models surpass diffusion models🚀:
<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/cc30b043-fa4e-4d01-a9b1-e50650d5675d" width=55%>
<p>





## Model zoo
We provide the pretrained checkpoints which can be downloaded from the following link [checkpoints](https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth):




## Training Scripts

You can train the base model with various targets and loss weighting:
```shell

CUDA_VISIBLE_DEVICES='2' torchrun --nproc_per_node 1 --master_port 15625  scripts/image_train.py  \
                            --data_dir data/celeba_hq_256 \
                            --attention_resolutions 16 --class_cond False \
                            --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear \
                            --num_channels 128 --num_head_channels 64 \
                            --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --lr 2e-5 --batch_size 8 \
                            --rescale_learned_sigmas True --p2_gamma 1 --p2_k 1 --log_dir logs/CeleAHQ/Debias \
                            --predict_xstart False --predict_v False \
```
A folder named `log_dir` will be created to save the checkpoints and logs.
The default setting is noise_prediction. You can specify the training targets to x0_prediction via setting `predict_xstart=True`， and v_prediction via `predict_v=True`.
The different noise weighting settings in noise_prediction mode are available in L863-882 in guided_diffusion/gaussion_diffusion.py


## Sampling Scripts
```shell

CUDA_VISIBLE_DEVICES='2' torchrun --nproc_per_node 1 --master_port 15625  image_sample.py \
                            --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 \
                            --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 \
                            --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 \
                            --predict_xstart True --predict_v False \
                            --model_path logs/CeleAHQ/debias_ema_0.9999_500000.pt \
                            --sample_dir samples/CelebAHQ_100/Debias \
                            --sampler ddpm
```
You can download our pretrained checkpoint to get for `model_path`. `timestep_respacing` specifies the sapling steps. You can also set the sampler as DDPM, DDIM, DPM-Solver, and UniPC. 


## Evaluation
The evaluation follows [guided_diffusion](https://github.com/openai/guided-diffusion)


## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Citation
If our work assists your research, feel free to give us a star ⭐ or cite us using:
```

@inproceedings{Debias,
      title={Unmasking Bias in Diffusion Model Training}, 
      author={Hu Yu and Li Shen and Jie Huang and Hongsheng Li and Feng Zhao},
      booktitle={The 18th European Conference on Computer Vision ECCV 2024},
      year={2024},
      organization={Springer}
}
```
