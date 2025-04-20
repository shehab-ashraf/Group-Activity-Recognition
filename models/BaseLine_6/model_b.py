import torch
import torch.nn as nn
import torchvision.models as models

class Group_Activity_Classifier(nn.Module):
    def __init__(self, Person_Activity_Model, hidden_size=512, num_classes=8):
        super(Group_Activity_Classifier, self).__init__()

        self.feature_extraction = nn.Sequential(*list(Person_Activity_Model.resnet50.children())[:-1])

        self.pool = nn.AdaptiveMaxPool2d((1, 2048))

        self.lstm = nn.LSTM(
            input_size=2048, 
            hidden_size=hidden_size, 
            num_layers=1, 
            batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        # Innput shape: (batch, seq, num_players, C, H, W)  
        batch, seq, num_players, C, H, W = x.size()
        # (batch, seq, num_players, C, H, W) --> (batch*seq*num_players, C, H, W)
        x = x.view(-1, C, H, W)
        # (batch*seq*num_players, C, H, W) --> (batch*seq*num_players, 2048)
        x = self.feature_extraction(x)
        # (batch*seq*num_players, 2048) --> (batch_size, seq, num_players, 2048)
        x = x.view(batch, seq, num_players, -1)
        # (batch, seq, num_players, 2048) --> (batch_size, seq, 2048)
        x = self.pool(x)
        # (batch, seq, 2048) --> (batch, seq, hidden_size)
        x, _ = self.lstm(x)
        # (batch, seq, hidden_size) --> (batch, hidden_size)
        x = x[:, -1, :]
        # (batch, hidden_size) --> (batch, num_classes)
        x = self.fc(x)
        return x
        
        
    
    
    