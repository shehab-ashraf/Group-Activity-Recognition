import torch
import torch.nn as nn
import torchvision.models as models

class Group_Activity_Classifier(nn.Module):
    def __init__(self, Person_Activity_Model, hidden_size=512, num_classes=8):
        super(Group_Activity_Classifier, self).__init__()

        self.resnet50 = nn.Sequential(*list(Person_Activity_Model.resnet50.children())[:-1])

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
            nn.Linear(2048, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        # Innput shape: (batch, seq, num_players, C, H, W)  
        batch, seq, num_players, C, H, W = x.size()
        # (batch, seq, num_players, C, H, W) --> (batch*seq*num_players, C, H, W)
        x = x.view(-1, C, H, W)
        # (batch*seq*num_players, C, H, W) --> (batch*seq*num_players, 2048)
        x = self.resnet50(x)
        # (batch*seq*num_players, 2048) --> (batch_size, seq, num_players, 2048)
        x1 = x.view(batch, seq, num_players, -1) # (batch, seq, num_players, 2048)
        x2 = x1.permute(0, 2, 1, 3).contiguous() # (batch_size, num_players, seq, 2048)
        x2 = x2.view(-1, seq, 2048) # (batch_size*num_players, seq, 2048)
        x2, _ = self.lstm(x2)
        x2 = x2.view(batch, num_players, seq, -1) # (batch_size, num_players, seq, 512)
        x2 = x2.permute(0, 2, 1, 3).contiguous() # (batch_size, seq, num_players, 512)
        x = torch.cat([x1, x2], dim=-1) # (batch_size, seq, num_players, 512+2048)
        # (batch, seq, num_players, 2048) --> (batch_size, seq, 2048)
        x = self.pool(x)
        x = x.squeeze(-2)
        # (batch, seq,  2048) --> (batch, 2048)
        x = x[:, -1, :]
        # (batch, 2048) --> (batch, num_classes)
        x = self.fc(x)
        return x