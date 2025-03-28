import torch
import torch.nn as nn

from typing import Optional, Dict, List, Tuple

from models.faster_rcnn import (
    FasterRCNN_ResNet50_FPN_Weights,
    FastRCNNPredictor,
    fasterrcnn_resnet50_fpn,
)


class Model(nn.Module):
    def __init__(self, num_classes: int):
        super(Model, self).__init__()

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
