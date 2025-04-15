import torch
import torch.nn as nn

class Group_Activity_Classifier(nn.Module):
    def __init__(self, Player_Activity_Model, num_classes=8, group_lstm_hidden_size=512):
        super().__init__()

        self.resnet = Player_Activity_Model.resnet
        self.player_lstm = Player_Activity_Model.lstm

        for param in Player_Activity_Model.parameters():
            param.requires_grad = False
            
        self.group_lstm = nn.LSTM(
            input_size=(2048+512),
            hidden_size=group_lstm_hidden_size,
            num_layers=1,
            batch_first=True
        )
        
        self.fc = nn.Sequential(
            nn.Linear(group_lstm_hidden_size, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
        
            nn.Linear(256, num_classes)
        )

        
    def forward(self, x):
        # Input shpae : batch, seq, num_players, C, H, W
        batch, seq, num_players, C, H, W = x.size()
        # (batch, seq, num_players, C, H, W) --> (batch*seq*num_players,  C, H, W)
        x = x.view(-1, C, H, W)
        # (batch*seq*num_players,  C, H, W) --> (batch*seq*num_players, 2048)
        x = self.resnet(x)
        # (batch*seq*num_players, 2048) --> (batch*num_players, seq, 2048+512)
        x1 = x.view(batch*num_players, seq, -1) # (batch*num_players, seq, 2048)
        x2 = self.player_lstm(x1) # (batch*num_players, seq, 512)
        x = torch.cat([x1, x2], dim=-1)

        # left_team, right_team
        x = x.view(batch, seq, num_players, -1) # (batch, seq, num_player, 2048+512)
        left_team = x[:, :, :6, :] # (btach, seq, 6, 2048+512)
        right_team = x[:, :, 6:, :] # (btach, seq, 6, 2048+512)
        left_team = torch.max(left_team, dim=2)[0] # (batch, seq, 2048+512)
        right_team = torch.max(right_team, dim=2)[0] # (batch, seq, 2048+512)
        x = torch.cat([left_team, right_team], dim=-1) # (batch, seq, 2*(2048+512))

        # (batch, seq, 2*(2048+512)) --> (batch, seq, group_lstm_hidden_size)
        x, _ = self.group_lstm(x)
        # (batch, seq, group_lstm_hidden_size) --> (batch, group_lstm_hidden_size)
        x = x[:, -1, :]
        # (batch, group_lstm_hidden_size) --> (batch, num_classes) 
        x = self.fc(x)
        return x