"""
the resnet model is fine-tuned to recognize person-level actions
"""
import torch.nn as nn
import torchvision.models as models

class Classifer(nn.Module):
    def __init__(self, num_classes):
        super(Classifer, self).__init__()
        
        self.resnet50 = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet50.fc = nn.Linear(in_features=self.resnet50.fc.in_features, out_features=num_classes)
    
    def forward(self, x):
        return self.resnet50(x)