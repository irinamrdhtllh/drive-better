import os
import argparse
import time
import cv2
import torch
import torch.optim as optim

from torch.utils.data import ConcatDataset

from carla_env.config import read_config
from carla_env.init import InitEnv
from datasets.dataset import RoadDamageDataset
from models.model import FasterRCNN_ResNet50, YOLO11
from scripts.train import train
from scripts.test import test
from utils import train_val_split, xml_to_yolotxt, visualize_boxes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="device to use the model",
    )

    parser.add_argument(
        "--model",
        type=str,
        choices=["frcnn", "yolo"],
        help="'frcnn': Faster R-CNN, 'yolo': YOLOv11",
    )

    subparsers = parser.add_subparsers(dest="command")
    train_subparser = subparsers.add_parser("train")
    test_subparser = subparsers.add_parser("test")
    inference_subparser = subparsers.add_parser("inference")

    return parser.parse_args()


def prepare_datasets(dataset_dir: str):
    countries = ["czech", "india", "japan", "norway", "united_states"]

    split_done = False
    conversion_done = False

    for country in countries:
        val_dir = os.path.join(dataset_dir, country, "val")
        if not os.path.exists(val_dir):
            train_val_split(dir=os.path.join(dataset_dir, country), val_ratio=0.2)
            split_done = True

        for split in ["train", "val"]:
            labels_dir = os.path.join(dataset_dir, country, split, "labels")
            if not os.path.exists(labels_dir):
                xml_to_yolotxt(dir=os.path.join(dataset_dir, country, split))
                conversion_done = True

    if split_done:
        print("Dataset train-val split completed.")
    else:
        print("Validation data already exists.")

    if conversion_done:
        print(f"Successfully converted XML annotation files into YOLO format.")
    else:
        print(f"Labels of data already exists.")


def load_datasets(dataset_dir: str, split: str = "train") -> ConcatDataset:
    countries = ["czech", "india", "japan", "norway", "united_states"]

    datasets = [
        RoadDamageDataset(dir=os.path.join(dataset_dir, country), split=split)
        for country in countries
    ]
    return ConcatDataset(datasets)


if __name__ == "__main__":
    args = parse_args()
    print(args)
    device = torch.device(args.device)
    dataset_dir = "./datasets/dataset"

    if args.command == "train":
        # Prepare the train and val datasets
        prepare_datasets(dataset_dir)

        # Load the datasets
        train_data = load_datasets(dataset_dir, split="train")
        val_data = load_datasets(dataset_dir, split="val")
        print("Dataset loaded successfully. Starting to train the model.")

        # Define the model
        if args.model == "frcnn":
            model = FasterRCNN_ResNet50().to(device)
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
                batch_size=4,
                image_size=640,
                device=device,
            )

        elif args.model == "yolo":
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

    elif args.command == "test":
        test_data = load_datasets(dataset_dir, split="test")

        print("Dataset loaded sucessfully. Starting to test the model.")

        if args.model == "frcnn":
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

        elif args.model == "yolo":
            model = YOLO11("./runs/detect/train/weights/best.pt")
            results = test(
                model,
                "./datasets/dataset/united_states/test/images/United_States_004805.jpg",
                device,
            )

            for result in results:
                boxes = result.boxes
                masks = result.masks
                keypoints = result.keypoints
                probs = result.probs
                obb = result.obb
                result.show()

    elif args.command == "inference":
        config = read_config()
        env = InitEnv(config)
        env.setup_experiment()
        env.reset_hero()
        env.generate_traffic()

        # Let actors finish spawning and physics settle
        for _ in range(10):
            env.tick(control=None)

        # Send camera image to the trained YOLO11 model to detect cracks and/or potholes
        sensor_data = env.get_sensor_data()
        camera_image = sensor_data["camera"][1]
        camera_image = cv2.cvtColor(camera_image, cv2.COLOR_RGB2BGR)

        model = YOLO11("./runs/detect/train/weights/best.pt")
        results = model.predict([camera_image])

        for result in results:
            boxes = result.boxes
            masks = result.masks
            keypoints = result.keypoints
            probs = result.probs
            obb = result.obb
            result.show()

        try:
            while True:
                env.tick(control=None)
                time.sleep(0.02)
        except KeyboardInterrupt:
            settings = env.world.get_settings()
            settings.synchronous_mode = False
            env.world.apply_settings(settings)
            env.destroy()
            time.sleep(0.5)
