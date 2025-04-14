import torch.nn as nn
import torchvision.models as models
import torch.nn.init as init

class Player_Activity_Classifier(nn.Module):
    def __init__(self, num_classes=9):
        super(Player_Activity_Classifier, self).__init__()
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Sequential(
            nn.Linear(self.resnet50.fc.in_features, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()


    def _initialize_weights(self):
        for m in self.resnet50.fc.modules():
            if isinstance(m, nn.Linear):
                init.xavier_normal_(m.weight, gain=1.0)
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            
    def forward(self, x):
        # Input shape: (batch_size, C, H, W)
        return self.resnet50(x)