import torch
import torch.nn as nn
from torchvision.models import resnet50

class Player_Activity_Classifier(nn.Module):
    def __init__(self, num_classes=9, lstm_hidden_size=512, lstm_num_layers=1):
        super().__init__()
        
        self.resnet = resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()
        
        self.lstm = nn.LSTM(
            input_size=2048,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(lstm_hidden_size+2048, lstm_hidden_size),
            nn.LayerNorm(lstm_hidden_size),
            nn.GELU(),
            nn.Dropout(0.3),
            
            nn.Linear(lstm_hidden_size, num_classes)
        )
    
    def forward(self, x):
        # Input shape: batch, seq, C, H, W
        batch, seq, C, H, W = x.size()
        # (batch, seq, C, H, W) --> (batch*seq, C, H, W)
        x = x.view(-1, C, H, W)
        # (batch*seq, C, H, W) --> (batch, seq, 2048+lstm_hidden_size)
        x1 = self.resnet(x) # (batch*seq, 2048)
        x1 = x1.view(batch, seq, -1) # (batch, seq, 2048)
        x2, _ = self.lstm(x1) # (batch, seq, lstm_hidden_size)
        x = torch.cat([x1, x2], dim=-1)
        # (batch, seq, 2048+lstm_hidden_size) --> (batch, 2048+lstm_hidden_size)
        x = x[:, -1, :]
        # (batch, 2048+lstm_hidden_size) --> (batch, num_classes)
        x = self.fc(x)
        return x




