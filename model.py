import torch
import torch.nn as nn
import torchvision.models as models


class Model(nn.Module):
    def __init__(self, num_classes: int):
        super(Model, self).__init__()

        self.model = models.detection.fasterrcnn_resnet50_fpn(num_classes=num_classes)

    def forward(self, images, targets=None):
        return self.model(images, targets)
