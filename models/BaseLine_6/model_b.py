import torch
import torch.nn as nn
import torchvision.models as models

class Group_Activity_Classifier(nn.Module):
    def __init__(self, Player_Activity_Model, hidden_size=512, num_classes=8):
        super(Group_Activity_Classifier, self).__init__()

        self.resnet50 = nn.Sequential(*list(Player_Activity_Model.resnet50.children())[:-1])

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
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Innput shape: (batch, seq, num_players, C, H, W)
        batch, seq, num_players, c, h, w = x.size()
        # (batch, seq, num_players, c, h, w) --> (batch*seq*num_players, c, h, w)
        x = x.view(-1, c, h, w)
        # (batch*seq*num_players, c, h, w) --> (batch*seq*num_players, 2048)
        x = self.resnet50(x)
        # (batch*seq*num_players, 2048) --> (batch, seq, num_players, 2048)
        x = x.view(batch, seq, num_players, -1)
        # (batch, seq, num_players, 2048) --> (batch_size, seq, 2048)
        x = self.pool(x)
        x = x.squeeze(-2)
        # (batch, seq, 2048) --> (batch, seq, hidden_size)
        x, _ = self.lstm(x)
        # (batch, seq, hidden_size) --> (batch, hidden_size)
        x = x[:, -1, :]
        # (batch, hidden_size) --> (batch, num_classes)
        x = self.fc(x)
        return x
        
        
    
    
    