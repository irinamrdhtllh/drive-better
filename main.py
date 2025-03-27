import os
import argparse
import torch
import torch.optim as optim

from torch.utils.data import ConcatDataset, random_split

from dataset import RoadDamageDataset
from model import Model
from train import train
from test import test
from utils import visualize_boxes


def parse_args() -> argparse.Namespace:
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

    return parser.parse_args()


def load_datasets(split: str = "train") -> ConcatDataset:
    countries = ["czech", "india", "japan", "norway", "united_states"]
    datasets = [
        RoadDamageDataset(dir=f"./dataset/{country}", split=split)
        for country in countries
    ]
    return ConcatDataset(datasets)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    if args.mode == "train":
        train_data = load_datasets(split="train")

        # Split into training and validation sets (80-20 split)
        train_size = int(0.8 * len(train_data))
        val_size = len(train_data) - train_size
        train_data, val_data = random_split(train_data, [train_size, val_size])

        print("Dataset loaded successfully. Starting to train the model.")

        model = Model(num_classes=5).to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.005)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        train(
            model,
            train_data,
            val_data,
            optimizer,
            lr_scheduler,
            num_epochs=100,
            device=device,
        )
    elif args.mode == "test":
        test_data = load_datasets(split="test")

        print("Dataset loaded sucessfully. Starting to test the model.")

        model = Model(num_classes=5).to(device)
        predictions, targets = test(model, test_data, device)

        # Average confidence score
        total_score = 0
        n_prediction = 0
        for prediction in predictions:
            for score in prediction["scores"]:
                total_score += score
                n_prediction += 1
        avg_score = total_score / n_prediction

        print(f"Avg confidence score: {avg_score:.4f}")

        # Visualize ground truth and predictions
        if targets is not None:
            sample_data = test_data.datasets[5]
            image_filename = sample_data.images[1] if sample_data.images else None
            if image_filename:
                image_path = os.path.join(sample_data.image_dir, image_filename)
                visualize_boxes(image_path, targets[1], predictions[1])
