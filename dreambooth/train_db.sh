export MODEL_NAME="CompVis/stable-diffusion-v1-4"
export INSTANCE_DIR="/home/siyuan/image_to_img_generation/resources/xray_images/normal"
export OUTPUT_DIR="./output/normal"

accelerate launch train_dreambooth.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="x-ray of healthy human chest" \
  --resolution=1024 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --learning_rate=5e-6 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=400 \
  --report_to="wandb"