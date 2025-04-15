import torch.nn as nn
import torchvision.models as models

class Player_Activity_Classifier(nn.Module):
    def __init__(self, num_classes=9):
        super(Player_Activity_Classifier, self).__init__()

        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)

        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.resnet50.fc.in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(1024, num_classes)
        )
            
    def forward(self, x):
        # Input shape: (batch, C, H, W)
        return self.resnet50(x)
    