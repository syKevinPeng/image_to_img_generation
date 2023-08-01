# Image to Image Generation

This repository contains code for image to image generation. The main work is based on Hugging Face's code for unconditional image generation.

## File Structure
```
.
├── .gitignore
├── baseline
│   ├── dataset.py
│   ├── model.py
│   └── train.py
├── hugging_face_img2img.ipynb
├── image_similarity_quality.ipynb
├── requirements.txt
└── uncond_img_gen
    ├── inference.ipynb
    ├── test.sh
    ├── train.sh
    └── train_unconditional.py
```

## File Descriptions

- `.gitignore`: Specifies intentionally untracked files that Git should ignore.
- `baseline/dataset.py`: Defines a PyTorch Dataset class for loading and transforming images.
- `baseline/model.py`: Defines a PyTorch Module class for the baseline model.
- `baseline/train.py`: Main training script for the baseline model.
- `hugging_face_img2img.ipynb`: Jupyter notebook for visualization and playgrounds.
- `image_similarity_quality.ipynb`: Jupyter notebook for evaluating the quality and similarity of generated images.
- `requirements.txt`: Lists the Python dependencies required for this project.
- `uncond_img_gen/inference.ipynb`: Jupyter notebook for performing inference with the unconditional image generation model.
- `uncond_img_gen/test.sh`: Example script for testing the unconditional image generation model.
- `uncond_img_gen/train.sh`: Example script for training the unconditional image generation model.
- `uncond_img_gen/train_unconditional.py`: Main training script for the unconditional image generation model.

## Usage

The main script for training the unconditional image generation model is `uncond_img_gen/train_unconditional.py`. It accepts the following command line arguments:

- `--model_name_or_path`: Path to pretrained model or model identifier from huggingface.co/models.
- `--dataset_name`: Name of the dataset to use (via the datasets library).
- `--output_dir`: The output directory where the model predictions and checkpoints will be written.
- `--do_train`: Whether to run training.
- `--do_eval`: Whether to run eval on the dev set.
- `--do_predict`: Whether to run predictions on the test set.
- `--per_device_train_batch_size`: Batch size per device during training.
- `--per_device_eval_batch_size`: Batch size for evaluation.
- `--learning_rate`: Learning rate for the optimizer.
- `--num_train_epochs`: Total number of training epochs to perform.

Example usage:

```bash
python train_unconditional.py --model_name_or_path="gpt2" --dataset_name="cifar10" --output_dir="./output" --do_train --do_eval --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --learning_rate=2e-5 --num_train_epochs=3
```

The `train.sh` and `test.sh` scripts in the `uncond_img_gen` directory provide example usage of the `train_unconditional.py` script for training and testing the model, respectively.