import os
import argparse
import torch
import torch.optim as optim

from torch.utils.data import random_split

from dataset import RoadDamageDataset
from model import Model
from train import train
from test import test
from utils import visualize_annotation

parser = argparse.ArgumentParser()
parser.add_argument(
    "--mode",
    type=str,
    choices=["train", "test"],
    required=True,
    help="Mode: 'train' or 'test'",
)
parser.add_argument(
    "--device",
    type=str,
    default="cuda" if torch.cuda.is_available() else "cpu",
    help="Device: 'cpu' or 'cuda'",
)

args = parser.parse_args()

device = torch.device(args.device)

model = Model(num_classes=5).to(device)
params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

if args.mode == "train":
    czech_train_data = RoadDamageDataset(dir="./dataset/czech", split="train")
    india_train_data = RoadDamageDataset(dir="./dataset/india", split="train")
    japan_train_data = RoadDamageDataset(dir="./dataset/japan", split="train")
    norway_train_data = RoadDamageDataset(dir="./dataset/norway", split="train")
    us_train_data = RoadDamageDataset(dir="./dataset/united_states", split="train")

    train_data = torch.utils.data.ConcatDataset(
        [
            czech_train_data,
            india_train_data,
            japan_train_data,
            norway_train_data,
            us_train_data,
        ]
    )
    train_size = int(0.8 * len(train_data))
    val_size = len(train_data) - train_size
    train_data, val_data = random_split(train_data, [train_size, val_size])

    print("Dataset parsed successfully. Starting training the model.")

    train(
        model,
        train_data,
        val_data,
        optimizer,
        lr_scheduler,
        num_epochs=100,
        device=device,
    )

if args.mode == "test":
    czech_test_data = RoadDamageDataset(dir="./dataset/czech", split="test")
    india_test_data = RoadDamageDataset(dir="./dataset/india", split="test")
    japan_test_data = RoadDamageDataset(dir="./dataset/japan", split="test")
    norway_test_data = RoadDamageDataset(dir="./dataset/norway", split="test")
    us_test_data = RoadDamageDataset(dir="./dataset/united_states", split="test")
    test_data = torch.utils.data.ConcatDataset(
        [
            czech_test_data,
            india_test_data,
            japan_test_data,
            norway_test_data,
            us_test_data,
        ]
    )

    predictions, targets = test(model, test_data, device)

    if targets is not None:
        dataset = us_test_data
        image_filename = dataset.images[1]
        image_path = os.path.join(dataset.image_dir, image_filename)
        annotation = dataset.annotations.get(image_filename, None)
        visualize_annotation(image_path, annotation)
