from http.client import ImproperConnectionState
import torch
from torch.utils.data import DataLoader
import wandb
import argparse
from dataset import XrayDataset




if __name__ == "__main__":
    # cmd line arguments
    parser = argparse.ArgumentParser()

    # dataset related arguments
    parser.add_argument("--dataset_path", type=str, default="../resources/xray_images")

    # training related arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--epochs", type=int, default=10)

    # wandb related arguments
    parser.add_argument("--wandb", type=bool, default=True)
    parser.add_argument("--wandb_exp_name", type=str, default="baseline")

    # exp related arguments
    parser.add_argument("--output_dir", type=str, default="./output")
    args = parser.parse_args()


    # Initialize wandb
    wandb.init(project="baseline",
            config=args,
            name=args.name,
            dir = args.output_dir)

