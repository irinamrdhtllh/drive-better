import torch
import torch.nn as nn

from typing import Optional, Dict, List, Tuple
from ultralytics import YOLO

from .faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
)


class FasterRCNN_ResNet50(nn.Module):
    def __init__(self, num_classes: int = 5):
        super(FasterRCNN_ResNet50, self).__init__()

        self.model = fasterrcnn_resnet50_fpn(
            weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )

        # Replace the pre-trained heads to predict 5 classes (4 road damage types + 1 background)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        return self.model(images, targets)


class YOLO11:
    def __init__(self):
        self.model = YOLO("yolo11n.pt")

    def train(
        self,
        data: str,
        num_epochs: int,
        batch_size: int,
        image_size: int,
        device: str,
        initial_lr: float,
    ):
        self.model.train(
            data=data,
            epochs=num_epochs,
            batch=batch_size,
            imgsz=image_size,
            device=device,
            lr0=initial_lr,
        )

    def evaluate(self):
        return self.model.val()

    def predict(self, image):
        return self.model(image)
