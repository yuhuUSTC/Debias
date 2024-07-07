# Unmasking Bias in Diffusion Model Training

<div align="center">

[![arXiv](https://img.shields.io/badge/arXiv%20paper-2404.02905-b31b1b.svg)](https://arxiv.org/abs/2404.02905)&nbsp;
[![huggingface weights](https://img.shields.io/badge/%F0%9F%A4%97%20Weights-FoundationVision/var-yellow)](https://huggingface.co/FoundationVision/var)&nbsp;




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


### 🔥 Discovering power-law Scaling Laws in VAR transformers📈:


<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/c35fb56e-896e-4e4b-9fb9-7a1c38513804" width=85%>
<p>
<p align="center">
<img src="https://github.com/FoundationVision/VAR/assets/39692511/91d7b92c-8fc3-44d9-8fb4-73d6cdb8ec1e" width=85%>
<p>



## Model zoo
We provide the pretrained checkpoints which can be downloaded from the following link [checkpoints](https://huggingface.co/FoundationVision/var/resolve/main/var_d16.pth):




## Training Scripts

You can train the base model with various targets and loss weighting:
```shell
# d16, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=16 --bs=768 --ep=200 --fp16=1 --alng=1e-3 --wpe=0.1
# d20, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=20 --bs=768 --ep=250 --fp16=1 --alng=1e-3 --wpe=0.1
# d24, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=24 --bs=768 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-4 --wpe=0.01
# d30, 256x256
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=30 --bs=1024 --ep=350 --tblr=8e-5 --fp16=1 --alng=1e-5 --wpe=0.01 --twde=0.08
# d36-s, 512x512 (-s means saln=1, shared AdaLN)
torchrun --nproc_per_node=8 --nnodes=... --node_rank=... --master_addr=... --master_port=... train.py \
  --depth=36 --saln=1 --pn=512 --bs=768 --ep=350 --tblr=8e-5 --fp16=1 --alng=5e-6 --wpe=0.01 --twde=0.08
```
A folder named `local_output` will be created to save the checkpoints and logs.
You can monitor the training process by checking the logs in `local_output/log.txt` and `local_output/stdout.txt`, or using `tensorboard --logdir=local_output/`.

If your experiment is interrupted, just rerun the command, and the training will **automatically resume** from the last checkpoint in `local_output/ckpt*.pth` (see [utils/misc.py#L344-L357](utils/misc.py#L344-L357)).

## Sampling & Zero-shot Inference

For FID evaluation, use `var.autoregressive_infer_cfg(..., cfg=1.5, top_p=0.96, top_k=900, more_smooth=False)` to sample 50,000 images (50 per class) and save them as PNG (not JPEG) files in a folder. Pack them into a `.npz` file via `create_npz_from_sample_folder(sample_folder)` in [utils/misc.py#L344](utils/misc.py#L360).
Then use the [OpenAI's FID evaluation toolkit](https://github.com/openai/guided-diffusion/tree/main/evaluations) and reference ground truth npz file of [256x256](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz) or [512x512](https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/512/VIRTUAL_imagenet512.npz) to evaluate FID, IS, precision, and recall.

Note a relatively small `cfg=1.5` is used for trade-off between image quality and diversity. You can adjust it to `cfg=5.0`, or sample with `autoregressive_infer_cfg(..., more_smooth=True)` for **better visual quality**.
We'll provide the sampling script later.

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
