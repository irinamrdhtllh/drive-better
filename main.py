import os
import argparse
import torch
import torch.optim as optim

from torch.utils.data import ConcatDataset, random_split

from datasets.dataset import RoadDamageDataset
from models.model import FasterRCNN_ResNet50, YOLO11
from scripts.train import train
from scripts.test import test
from utils import train_val_split, xml_to_yolotxt, visualize_boxes


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


def split_datasets():
    dataset_dir = "./datasets/dataset"
    countries = ["czech", "india", "japan", "norway", "united_states"]

    split = False

    for country in countries:
        val_dir = os.path.join(dataset_dir, country, "val")
        if not os.path.exists(val_dir):
            train_val_split(dir=f"./datasets/dataset/{country}", val_ratio=0.2)
            split = True

    if split:
        print("Dataset train-val split completed.")
    else:
        print("Validation data already exists.")


def load_datasets(split: str = "train") -> ConcatDataset:
    countries = ["czech", "india", "japan", "norway", "united_states"]

    convert = False

    for country in countries:
        labels_dir = f"./datasets/dataset/{country}/{split}/labels"
        if not os.path.exists(labels_dir):
            xml_to_yolotxt(dir=f"./datasets/dataset/{country}/{split}")
            convert = True

    if convert:
        print(f"Successfully converted {split} XML annotation files into YOLO format.")
    else:
        print(f"Labels of {split} data already exists.")

    datasets = [
        RoadDamageDataset(dir=f"./datasets/dataset/{country}", split=split)
        for country in countries
    ]
    return ConcatDataset(datasets)


if __name__ == "__main__":
    args = parse_args()
    device = torch.device(args.device)

    if args.mode == "train":
        # Split into training and validation sets (80-20 split)
        split_datasets()

        # Load the datasets
        train_data = load_datasets(split="train")
        val_data = load_datasets(split="val")
        print("Dataset loaded successfully. Starting to train the model.")

        # Define the model
        # model = FasterRCNN_ResNet50().to(device)
        # params = [p for p in model.parameters() if p.requires_grad]
        # optimizer = optim.SGD(params, lr=0.01, momentum=0.9, weight_decay=0.005)
        # lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.9)

        model = YOLO11()

        train(
            model,
            train_data,
            val_data,
            num_epochs=100,
            batch_size=4,
            image_size=640,
            device=device,
        )
    elif args.mode == "test":
        test_data = load_datasets(split="test")

        print("Dataset loaded sucessfully. Starting to test the model.")

        model = FasterRCNN_ResNet50().to(device)
        predictions, targets = test(model, test_data, device)

        # Average confidence score
        total_score = 0
        num_predictions = 0
        for prediction in predictions:
            for score in prediction["scores"]:
                total_score += score
                num_predictions += 1
        avg_score = total_score / num_predictions

        print(f"Avg confidence score: {avg_score:.4f}")

        # Visualize ground truth and predictions
        if targets is not None:
            sample_data = test_data.datasets[5]
            image_filename = sample_data.images[1] if sample_data.images else None
            if image_filename:
                image_path = os.path.join(sample_data.image_dir, image_filename)
                visualize_boxes(image_path, targets[1], predictions[1])
