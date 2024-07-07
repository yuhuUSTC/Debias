export PYTHONPATH=$PYTHONPATH:/mnt/workspace/workgroup/yuhu/code/Debias

## Single class datasets  --  FFHQ, AFHQ, CelebAHQ, MetFaces
CUDA_VISIBLE_DEVICES='2' torchrun --nproc_per_node 1 --master_port 15625  /mnt/workspace/workgroup/yuhu/code/Debias/scripts/image_sample.py \
                            --attention_resolutions 16 --class_cond False --diffusion_steps 1000 --dropout 0.0 --image_size 256 \
                            --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 \
                            --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 \
                            --predict_xstart True --predict_v False \
                            --model_path /mnt/workspace/workgroup/yuhu/code/Debias/logs/CeleAHQ/start/ema_0.9999_500000.pt \
                            --sample_dir /mnt/workspace/workgroup/yuhu/code/Debias/samples/CelebAHQ_100/Start \
                            --sampler ddpm
                            
## Multi class dataset  --  CIFAR10       
# CUDA_VISIBLE_DEVICES='6' torchrun --nproc_per_node 1 --master_port 15665  /mnt/workspace/workgroup/yuhu/code/Debias/scripts/image_sample.py \
#                             --attention_resolutions 16 --class_cond True --num_classes 10 --diffusion_steps 1000 --dropout 0.0 --image_size 32 \
#                             --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 \
#                             --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 1000 \
#                             --predict_xstart False --predict_v False \
#                             --model_path /mnt/workspace/workgroup/yuhu/code/Debias/logs/CIFAR10v1/Constant/ema_0.9999_500000.pt \
#                             --sample_dir /mnt/workspace/workgroup/yuhu/code/Debias/samples/CIFAR10/Constant1/1000 \
#                             --sampler ddpm

## Multi class dataset  --  ImageNet
# CUDA_VISIBLE_DEVICES='2' torchrun --nproc_per_node 1 --master_port 15625  /mnt/workspace/workgroup/yuhu/code/Debias/scripts/image_sample.py \
#                             --attention_resolutions 16 --class_cond True --num_classes 1000 --diffusion_steps 1000 --dropout 0.0 --image_size 256 \
#                             --learn_sigma True --noise_schedule linear --num_channels 128 --num_res_blocks 1 --num_head_channels 64 \
#                             --resblock_updown True --use_fp16 False --use_scale_shift_norm True --timestep_respacing 100 \
#                             --predict_xstart False --predict_v True \
#                             --model_path /mnt/workspace/workgroup/yuhu/code/Debias/logs/ImageNet/Vol/ema_0.9999_500000.pt \
#                             --sample_dir /mnt/workspace/workgroup/yuhu/code/Debias/samples/ImageNet/100/Vol
#                             --sampler ddpm