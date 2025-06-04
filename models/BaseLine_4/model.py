import torch.nn as nn
import torchvision.models as models

class Group_Activity_Classifier(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=1, num_classes=8):
        super(Group_Activity_Classifier, self).__init__()

        self.resnet50 = nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1])  
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3), 

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(128, num_classes)  
        )

    def forward(self, x):
        # Input_Shape: (Batch_Size, 9, 3, 224, 224)
        batch, seq, c, h, w = x.shape
        # (batch, seq, c, h, w) --> (batch*seq, c, h, w)
        x = x.view(batch * seq, c, h, w)  
        # (batch*seq, c, h, w) --> (batch*seq, 2048)
        x = self.resnet50(x)
        # (batch*seq, 2048) --> (batch, seq, 2048)
        x = x.view(batch, seq, -1)
        # (batch, seq, 2048) --> (batch, seq, hidden_dim)
        x, _ = self.lstm(x)
        # (batch, seq, hidden_dim) --> (batch, hidden_dim)
        x = x[:, -1, :]
        # (batch, hidden_dim) --> (batch, num_classes)
        x = self.fc(x)
        return x
