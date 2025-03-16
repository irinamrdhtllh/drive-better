import torch
import torch.nn as nn

from faster_rcnn import fasterrcnn_resnet50_fpn


class Model(nn.Module):
    def __init__(self, num_classes: int):
        super(Model, self).__init__()

        self.model = fasterrcnn_resnet50_fpn(num_classes=num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
