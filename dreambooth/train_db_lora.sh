export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export INSTANCE_DIR="/home/siyuan/research/image_to_img_generation/resources/xray_images/normal"
export OUTPUT_DIR="output_lora"

accelerate launch train_dreambooth_lora.py \
  --pretrained_model_name_or_path=$MODEL_NAME  \
  --instance_data_dir=$INSTANCE_DIR \
  --output_dir=$OUTPUT_DIR \
  --instance_prompt="x-ray of human chest" \
  --resolution=512 \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --checkpointing_steps=100 \
  --learning_rate=1e-4 \
  --report_to="wandb" \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_train_steps=500 \
  --validation_prompt="x-ray of human chest" \
  --validation_epochs=5 \
  --seed="0" 