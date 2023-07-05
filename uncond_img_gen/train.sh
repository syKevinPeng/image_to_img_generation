accelerate launch train_unconditional.py \
  --train_data_dir="/home/siyuan/image_to_img_generation/resources/xray_images/gi" \
  --num_img_to_train 10 \
  --resolution 512 \
  --output_dir "./output_gi" \
  --train_batch_size=1 \
  --num_epochs 1200 \
  --gradient_accumulation_steps=1 \
  --learning_rate=1e-4 \
  --lr_warmup_steps=500 \
  --mixed_precision=no \
  --eval_batch_size 4 \
  --save_images_epochs 1200 \
  --ddpm_num_steps 1000 \
  --ddpm_num_inference_steps 1000 \
  --save_model_epochs 10000 \
  --enable_xformers_memory_efficient_attention \