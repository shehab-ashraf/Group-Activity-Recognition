import torch
import torch.nn as nn

class GroupActivityClassifier(nn.Module):
    def __init__(self, model):
        super(GroupActivityClassifier, self).__init__()

        self.feature_extractor = torch.nn.Sequential(*list(model.resnet50.children())[:-1])
        self.feature_extractor.requires_grad_(False) 

        self.pool = nn.AdaptiveMaxPool2d((1, 2048))

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 8)
        )

    def forward(self, x):
        # (batch, bbox, C, H, W) --> (batch * bbox, C, H, W)
        batch, bbox, C, H, W = x.size()
        x = x.view(batch * bbox, C, H, W)

        # (batch * bbox, C, H, W) --> (batch, bbox, 2048)
        x = self.feature_extractor(x)
        x = x.view(batch, bbox, -1) 

        # (batch, bbox, 2048) --> (batch, 2048)
        x = self.pool(x) 
        x = x.squeeze()

        # (batch, 2048) --> (batch, 8)
        x = self.fc(x)

        return x
