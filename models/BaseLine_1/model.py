import torch.nn as nn
from torchvision import models
from typing import Optional

class GroupActivityClassifier(nn.Module):
    def __init__(self, num_classes: int, freeze_backbone: Optional[bool] = False):
        super(GroupActivityClassifier, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)

        if freeze_backbone:
            for param in self.resnet50.parameters():
                param.requires_grad = False
        
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)