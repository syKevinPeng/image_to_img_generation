# This script is used to generate images from trained weights
accelerate launch train_unconditional.py \
    --test \
    --num_img_to_gen 10 \
    --resolution 512 \
    --output_dir "./output_gen" \
    --eval_batch_size 4 \
    --ddpm_num_steps 1000 \
    --ddpm_num_inference_steps 1000 \
    --resume_from_checkpoint "/home/siyuan/image_to_img_generation/uncond_img_gen/output_pacemaker/checkpoint-12000" \
