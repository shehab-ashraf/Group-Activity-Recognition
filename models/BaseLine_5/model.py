import torch
import torch.nn as nn
import torchvision.models as models

class Group_Activity_Classifier(nn.Module):
    def __init__(self, hidden_size=512, num_classes=8):
        super(Group_Activity_Classifier, self).__init__()

        self.resnet50 = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1])

        for param in self.resnet50.parameters():
            param.requires_grad = False
        
        self.pool = nn.AdaptiveMaxPool2d((1, 2048))

        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Input_Shape: (batch, seq, num_players, c, h, w)
        batch, seq, num_players, c, h, w = x.size()
        # (batch, seq, num_players, c, h, w) --> (batch*seq*num_players, c, h, w)
        x = x.view(-1, c, h, w)
        # (batch*seq*num_players, c, h, w) --> (batch*seq*num_players, 2048)
        x = self.resnet50(x)
        # (batch*seq*num_players, 2048) --> (batch, seq, num_players, 2048)
        x = x.view(batch, seq, num_players, -1)
        # (batch, seq, num_players, 2048) --> (batch, seq, 2048)
        x = self.pool(x)
        x = x.squeeze(-2)
        # (batch, seq, 2048) --> (batch, seq, hidden_size)
        x, _ = self.lstm(x)
        # (batch, seq, hidden_size) --> (batch, hidden_size)
        x = x[:, -1, :]
        # (batch, hidden_size) --> (batch, num_classes)
        x = self.fc(x)
        return x
        
        
    
    
    