import torch
import torch.optim as optim

from typing import Optional, Dict, List, Tuple, Union
from torch.utils.data import DataLoader
from torch.optim import Optimizer, lr_scheduler
from tqdm import tqdm

from models.model import FasterRCNN_ResNet50, YOLO11
from utils import collate_fn


def train(
    model: Union[FasterRCNN_ResNet50, YOLO11],
    train_data: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    val_data: List[Tuple[torch.Tensor, Dict[str, torch.Tensor]]],
    optimizer: Optional[Optimizer] = None,
    lr_scheduler: Optional[
        Union[lr_scheduler._LRScheduler, lr_scheduler.ReduceLROnPlateau]
    ] = None,
    num_epochs: int = 100,
    batch_size: int = 4,
    image_size: int = 640,
    device: str = "cpu",
):
    if isinstance(model, FasterRCNN_ResNet50):
        train_loader = DataLoader(
            train_data,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=collate_fn,
        )

        val_loader = DataLoader(
            val_data,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

        # Define the model
        model.to(device)

        # Training loop
        best_val_loss = float("inf")
        best_val_score = 0.0
        best_epoch = 0
        for epoch in range(num_epochs):
            model.train()
            total_train_loss = 0

            train_bar = tqdm(
                train_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Training]",
                leave=False,
            )

            for images, targets in train_bar:
                images = [i.to(device) for i in images]
                targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                # Forward pass
                losses, _ = model(images, targets)
                train_loss = sum(l for l in losses.values())

                # Backpropagation
                optimizer.zero_grad()
                train_loss.backward()
                optimizer.step()

                total_train_loss += train_loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # Validate the model
            model.eval()
            total_val_loss = 0
            total_val_score = 0
            num_predictions = 0

            val_bar = tqdm(
                val_loader,
                desc=f"Epoch {epoch + 1}/{num_epochs} [Validation]",
                leave=False,
            )

            with torch.no_grad():
                for images, targets in val_bar:
                    images = [i.to(device) for i in images]
                    targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

                    losses, predictions = model(images, targets)
                    val_loss = sum(l for l in losses.values())
                    total_val_loss += val_loss.item()

                    for prediction in predictions:
                        for score in prediction["scores"]:
                            total_val_score += score
                            num_predictions += 1

            avg_val_loss = total_val_loss / len(val_loader)

            avg_val_score = (
                total_val_score / num_predictions if num_predictions > 0 else 0.0
            )

            if lr_scheduler:
                if isinstance(lr_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    lr_scheduler.step(total_val_loss)
                else:
                    lr_scheduler.step()

            print(
                f"Epoch: [{epoch + 1}/{num_epochs}], Train Loss: {(avg_train_loss):.4f}, Val Loss: {avg_val_loss:.4f}, Val Score: {avg_val_score:.4f}"
            )

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_val_score = avg_val_score
                best_epoch = epoch + 1
                torch.save(model.state_dict(), "./road_damage_detector.pth")
                print(
                    f"Model saved at epoch {epoch + 1} with validation loss: {best_val_loss:.4f}."
                )

        print(
            f"Training complete. Best val loss: {best_val_loss:.4f} at epoch {best_epoch} with score: {best_val_score:.4f}."
        )

    elif isinstance(model, YOLO11):
        model.train(
            data="./datasets/dataset.yaml",
            num_epochs=num_epochs,
            batch_size=batch_size,
            image_size=image_size,
            device=device,
            initial_lr=0.001,
        )

    else:
        raise ValueError("Model type must be either PyTorch model or YOLO model.")
