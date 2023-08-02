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

Here are the command line arguments in the script along with their definitions:

1. `--dataset_name`: The name of the Dataset (from the HuggingFace hub) to train on. It can also be a path pointing to a local copy of a dataset in your filesystem, or to a folder containing files that HF Datasets can understand.

2. `--dataset_config_name`: The config of the Dataset, leave as None if there's only one config.

3. `--train`: Whether to run training.

4. `--test`: Whether to run testing and generate images.

5. `--num_img_to_train`: The number of images to use for training. If not set, all the images in the dataset will be used.

6. `--num_img_to_gen`: The number of images to generate for evaluation only. If not set, 4 images will be generated.

7. `--model_config_name_or_path`: The config of the UNet model to train, leave as None to use standard DDPM configuration.

8. `--train_data_dir`: A folder containing the training data. Folder contents must follow the structure described in https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file must exist to provide the captions for the images. Ignored if `dataset_name` is specified. 

9. `--output_dir`: The output directory where the model predictions and checkpoints will be written.

10. `--overwrite_output_dir`: Overwrite the content of the output directory.

11. `--cache_dir`: The directory where the downloaded models and datasets will be stored.

12. `--resolution`: The resolution for input images, all the images in the train/validation dataset will be resized to this resolution.

13. `--center_crop`: Whether to center crop the input images to the resolution. If not set, the images will be randomly cropped. The images will be resized to the resolution first before cropping.

14. `--random_flip`: Whether to randomly flip images horizontally.

15. `--train_batch_size`: Batch size (per device) for the training dataloader.

16. `--eval_batch_size`: The number of images to generate for evaluation.

17. `--dataloader_num_workers`: The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.

18. `--init_epochs`: The initial number of epochs to train the model.

19. `--save_images_epochs`: How often to save images during training.

20. `--save_model_epochs`: How often to save the model during training.

21. `--gradient_accumulation_steps`: Number of updates steps to accumulate before performing a backward/update pass.

22. `--learning_rate`: Initial learning rate (after the potential warmup period) to use.

23. `--lr_scheduler`: The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"].

24. `--lr_warmup_steps`: Number of steps for the warmup in the lr scheduler.

25. `--adam_beta1`: The beta1 parameter for the Adam optimizer.

26. `--adam_beta2`: The beta2 parameter for the Adam optimizer.

27. `--adam_weight_decay`: Weight decay magnitude for the Adam optimizer.

28. `--adam_epsilon`: Epsilon value for the Adam optimizer.

29. `--use_ema`: Whether to use Exponential Moving Average for the final model weights.

30. `--ema_inv_gamma`: The inverse gamma value for the EMA decay.

31. `--ema_power`: The power value for the EMA decay.

32. `--ema_max_decay`: The maximum decay magnitude for EMA.

33. `--push_to_hub`: Whether or not to push the model to the Hub.

34. `--hub_token`: The token to use to push to the Model Hub.

35. `--hub_model_id`: The name of the repository to keep in sync with the local `output_dir`.

36. `--hub_private_repo`: Whether or not to create a private repository.

37. `--logger`: Whether to use [tensorboard](https://www.tensorflow.org/tensorboard) or [wandb](https://www.wandb.ai) for experiment tracking and logging of model metrics and model checkpoints.

38. `--logging_dir`: [TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***.

39. `--local_rank`: For distributed training: local_rank.

40. `--mixed_precision`: Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10. and an Nvidia Ampere GPU.

41. `--prediction_type`: Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.

42. `--ddpm_num_steps`: Number of steps for the DDPM (Number of steps to add and recover from noise in training).

43. `--ddpm_num_inference_steps`: Number of inference steps for the DDPM.

44. `--ddpm_beta_schedule`: Beta schedule for the DDPM.

45. `--checkpointing_steps`: Save a checkpoint of the training state every X updates. These checkpoints are only suitable for resuming training using `--resume_from_checkpoint`.

46. `--checkpoints_total_limit`: Max number of checkpoints to store. Passed as `total_limit` to the `Accelerator` `ProjectConfiguration`.

47. `--resume_from_checkpoint`: Whether training should be resumed from a previous checkpoint. Use a path saved by `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.

48. `--enable_xformers_memory_efficient_attention`: Whether or not to use xformers.

49. `--target_fid`: The target FID score to reach before stopping training.

Note: During the training, the model will first train for fix number of epochs (as defined by "--init_epochs"). The model will stop if the model reaches target FID score \


## Example usage:

The `train.sh` and `test.sh` scripts in the `uncond_img_gen` directory provide example usage of the `train_unconditional.py` script for training and testing the model, respectively.