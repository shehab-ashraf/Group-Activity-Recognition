import torch
import torch.nn as nn

class Group_Activity_Classifier(nn.Module):
    def __init__(self, Player_Activity_Model, num_classes=8):
        super(Group_Activity_Classifier, self).__init__()

        self.resnet50 = torch.nn.Sequential(*list(Player_Activity_Model.resnet50.children())[:-1])

        for param in self.resnet50.parameters():
            param.requires_grad = False
            
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))

        self.fc = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

    def forward(self, x):
        # (batch, bbox, C, H, W) --> (batch * bbox, C, H, W)
        batch, bbox, C, H, W = x.size()
        x = x.view(batch * bbox, C, H, W)
        # (batch * bbox, C, H, W) --> (batch, bbox, 2048)
        x = self.resnet50(x)
        x = x.view(batch, bbox, -1)
        # (batch, bbox, 2048) --> (batch, 2048)
        x = self.pool(x) 
        x = x.squeeze()
        # (batch, 2048) --> (batch, 8)
        x = self.fc(x)
        return x
