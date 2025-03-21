"""

"""
import torch
import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self, model):
        super(Classifier, self).__init__()

        self.feature_extractor = torch.nn.Sequential(*list(model.resnet50.children())[:-1])
        self.feature_extractor.requires_grad_(False)

        self.pool = nn.AdaptiveMaxPool2d((1, 2048))
        

        self.fc = nn.Linear(in_features=2048, out_features=1024)
        self.batch_norm = nn.BatchNorm1d(1024)
        self.act = nn.ReLU()
        
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(in_features=1024, out_features=8)

    def forward(self, x):
        batch, bbox, C, H, W = x.size() # Input shape: (batch, bbox, C, H, W) 
        x = x.view(batch * bbox, C, H, W) # Reshape input to merge batch and bbox dimensions for feature extraction
        x = self.feature_extractor(x) # Pass through the feature extractor

        x = x.view(batch, bbox, -1)
        x = self.pool(x)

        x = x.squeeze(dim=1)  # Shape: (batch, 2048)
        x = self.fc(x)
        x = self.batch_norm(x)
        x = self.act(x)

        x = self.dropout(x)
        x = self.fc2(x)

        return x
        
