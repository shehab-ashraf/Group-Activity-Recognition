import torch
import torch.nn as nn

class GroupActivityClassifier(nn.Module):
    def __init__(self, model):
        super(GroupActivityClassifier, self).__init__()

        # Feature Extractor: ResNet50 (excluding last FC layer)
        self.feature_extractor = torch.nn.Sequential(*list(model.resnet50.children())[:-1])
        self.feature_extractor.requires_grad_(False) 

        self.bbox_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 8)
        )

    def forward(self, x):
        batch, bbox, C, H, W = x.size()
        x = x.view(batch * bbox, C, H, W)  # Flatten batch & bbox for ResNet

        x = self.feature_extractor(x)  # Extract features -> (batch * bbox, 2048, 1, 1)
        x = x.view(batch, bbox, -1)  # Reshape to (batch, bbox, 2048)

        x = self.bbox_pool(x.permute(0, 2, 1))  # (batch, 2048, 1)
        x = x.squeeze(-1)  # (batch, 2048)

        x = self.classifier(x)

        return x
