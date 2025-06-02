import torch.nn as nn
from torchvision import models
from typing import Optional

class Group_Activity_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Group_Activity_Classifier, self).__init__()

        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT) 
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, num_classes)

    def forward(self, x):
        # Input_Shape: (Batch_Size, 3, 224, 224)
        return self.resnet50(x)