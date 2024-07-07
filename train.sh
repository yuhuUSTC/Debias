export PYTHONPATH=$PYTHONPATH:/mnt/workspace/workgroup/yuhu/code/Debias

## Single class datasets  --  FFHQ, AFHQ, CelebAHQ, MetFaces
CUDA_VISIBLE_DEVICES='2' torchrun --nproc_per_node 1 --master_port 15625  /mnt/workspace/workgroup/yuhu/code/Debias/scripts/image_train.py  \
                            --data_dir /mnt/workspace/workgroup/yuhu/data/celeba_hq_256 \
                            --attention_resolutions 16 --class_cond False \
                            --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear \
                            --num_channels 128 --num_head_channels 64 \
                            --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --lr 2e-5 --batch_size 8 \
                            --rescale_learned_sigmas True --p2_gamma 1 --p2_k 1 --log_dir /mnt/workspace/workgroup/yuhu/code/Debias/logs/CeleAHQ/Debias \
                            --predict_xstart False --predict_v False \

## Multi class dataset  --  CIFAR10               
# CUDA_VISIBLE_DEVICES='3' torchrun --nproc_per_node 1 --master_port 15635  /mnt/workspace/workgroup/yuhu/code/Debias/scripts/image_train.py  \
#                             --data_dir /mnt/workspace/workgroup/yuhu/data/CIFAR10 \
#                             --attention_resolutions 16 --class_cond True  --num_classes 10 \
#                             --diffusion_steps 1000 --dropout 0.0 --image_size 32 --learn_sigma True --noise_schedule linear \
#                             --num_channels 128 --num_head_channels 64 \
#                             --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --lr 2e-5 --batch_size 32 \
#                             --rescale_learned_sigmas True --p2_gamma 1 --p2_k 1 --log_dir /mnt/workspace/workgroup/yuhu/code/Debias/logs/CIFAR10v1/Debias  \
#                             --predict_xstart False --predict_v False \           
                              
## Multi class dataset  --  ImageNet
# CUDA_VISIBLE_DEVICES='5' torchrun --nproc_per_node 1 --master_port 15655  /mnt/workspace/workgroup/yuhu/code/Debias/scripts/image_train.py  \
#                             --data_dir /earth-nas/datasets/imagenet1k/train \
#                             --attention_resolutions 16 --class_cond True  --num_classes 1000\
#                             --diffusion_steps 1000 --dropout 0.0 --image_size 256 --learn_sigma True --noise_schedule linear \
#                             --num_channels 128 --num_head_channels 64 \
#                             --num_res_blocks 1 --resblock_updown True --use_fp16 False --use_scale_shift_norm True --lr 2e-5 --batch_size 8 \
#                             --rescale_learned_sigmas True --p2_gamma 1 --p2_k 1 --log_dir /mnt/workspace/workgroup/yuhu/code/Debias/logs/ImageNet/Debias   \
#                             --predict_xstart False --predict_v False \