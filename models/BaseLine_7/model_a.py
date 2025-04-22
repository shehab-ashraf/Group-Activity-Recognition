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
            nn.Linear(lstm_hidden_size, 265),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            
            nn.Linear(256, num_classes)
        )
    
    
    def forward(self, x):
        # Input shape: batch, seq, C, H, W
        batch, seq, C, H, W = x.size()
        # (batch, seq, C, H, W) --> (batch*seq, C, H, W)
        x = x.view(-1, C, H, W)
        # (batch*seq, C, H, W) --> (batch, seq, lstm_hidden_size)
        x = self.resnet(x) # (batch*seq, 2048)
        x = x.view(batch, seq, -1) # (batch, seq, 2048)
        x, _ = self.lstm(x) # (batch, seq, lstm_hidden_size)
        # (batch, seq, lstm_hidden_size) --> (batch, lstm_hidden_size)
        x = x[:, -1, :]
        # (batch, lstm_hidden_size) --> (batch, num_classes)
        x = self.fc(x)
        return x




