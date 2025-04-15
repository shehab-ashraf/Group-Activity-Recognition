import torch
import torch.nn as nn

class Group_Activity_Classifier(nn.Module):
    def __init__(self, Player_Activity_Model, num_classes=8):
        super().__init__()

        self.resnet = Player_Activity_Model.resnet
        self.lstm = Player_Activity_Model.lstm

        for param in Player_Activity_Model.parameters():
            param.requires_grad = False
        
        self.fc = nn.Sequential(
            nn.Linear(2048+512, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(0.3),
        
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        # Input shpae: batch, seq, num_players, C, H, W
        batch, seq, num_players, C, H, W = x.size()
        # (batch, seq, num_players, C, H, W) --> (batch*seq*num_players, C, H, W)
        x = x.view(-1, C, H, W)
        # (batch*seq*num_players, C, H, W) --> (batch*seq*num_players, 2048)
        x1 = self.resnet(x)
        # (batch*seq*num_players, 2048) --> (batch*num_players, seq, 2048+512)
        x1 = x1.view(-1, seq, 2048) # (batch*num_players, seq, 2048)
        x2, _ = self.lstm(x1) # (batch*num_players, seq, 512)
        x = torch.cat([x1, x2], dim=-1) # (batch*num_players, seq, 2048+512)
        x = x.contiguous()
        # (batch*num_players, seq, 2048+512) --> (batch, seq, num_players, 2048+512)
        x = x.view(batch, seq, num_players, -1)
        # (batch, seq, num_players, 2048+512) --> # (batch, seq, 2048+512)
        x = torch.max(x, dim=2)[0]
        # (batch, seq, 2048+512) --> (batch, 2048+512)
        x = x[:, -1, :]
        # (batch, 2048+512) --> (batch, num_classes)
        x = self.fc(x)
        return x