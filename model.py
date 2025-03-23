import torch
import torch.nn as nn

from typing import Optional, Dict, List, Tuple

from models.faster_rcnn import fasterrcnn_resnet50_fpn


class Model(nn.Module):
    def __init__(self, num_classes: int):
        super(Model, self).__init__()

        self.model = fasterrcnn_resnet50_fpn(num_classes=num_classes)

    def forward(
        self,
        images: List[torch.Tensor],
        targets: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        return self.model(images, targets)
