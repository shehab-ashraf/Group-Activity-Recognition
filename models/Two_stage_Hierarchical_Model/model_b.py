import torch
import torch.nn as nn
import torchvision.models as models

class Group_Activity_Classifier(nn.Module):
    def __init__(self, Player_Activity_Model, hidden_size=512, num_classes=8):
        super(Group_Activity_Classifier, self).__init__()

        self.resnet50 = nn.Sequential(*list(Player_Activity_Model.resnet50.children())[:-1])

        for param in self.resnet50.parameters():
            param.requires_grad = False


        self.pool = nn.AdaptiveMaxPool2d((1, 1024))

        self.player_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )

        self.group_lstm = nn.LSTM(
            input_size=2048,
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(512, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.5),

            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        # Innput shape: (batch, seq, num_players, c, h, w)  
        batch, seq, num_players, c, h, w = x.size()
        # (batch, seq, num_players, c, h, w) --> (batch*seq*num_players, c, h, w)
        x = x.view(-1, c, h, w)
        # (batch*seq*num_players, c, h, w) --> (batch*seq*num_players, 2048)
        x = self.resnet50(x)

        # (batch*seq*num_players, 2048) --> (batch, seq, num_players, 2048)
        x1 = x.view(batch, seq, num_players, -1) # (batch, seq, num_players, 2048)
        x2 = x1.permute(0, 2, 1, 3).contiguous() # (batch, num_players, seq, 2048)
        x2 = x2.view(-1, seq, 2048) # (batch*num_players, seq, 2048)
        x2, _ = self.Player_lstm(x2)
        x2 = x2.view(batch, num_players, seq, -1) # (batch, num_players, seq, 512)
        x2 = x2.permute(0, 2, 1, 3).contiguous() # (batch, seq, num_players, 512)
        x = torch.cat([x1, x2], dim=-1) # (batch, seq, num_players, 2048+512)

        # left_team, right_team
        left_team = x[:, :, :6, :]  # (batch, seq, 6, 2048+512)
        right_team = x[:, :, 6:, :] # (batch, seq, 6, 2048+512)

        # pooling
        left_team = self.pool(left_team).squeeze(-2) # (batch, seq, 1024)
        right_team = self.pool(right_team).squeeze(-2) # (batch, seq, 1024)
        x = torch.cat([left_team, right_team], dim=-1) # (batch, seq, 2048)

        # (batch, seq, 2048) --> (batch, seq, 512)
        x, _ = self.Group_lstm(x)
        # (batch, seq,  512) --> (batch, 512)
        x = x[:, -1, :]
        # (batch, 512) --> (batch, num_classes)
        x = self.fc(x)
        return x