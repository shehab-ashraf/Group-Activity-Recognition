"""
This BaseLine is a temporal extension of the first BaseLine. It examines the idea of feeding
image level features directly to an LSTM model to recognize the group activity. In this baseline, the 
resnet50 model is deployed on the whole image and resulting AvgPool features are fed to an LSTM model.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class ResNetLSTM(nn.Module):
    def __init__(self, hidden_dim=512, num_layers=1, num_classes=None):
        super(ResNetLSTM, self).__init__()

        self.resnet = models.resnet50(pretrained=True)
        self.resnet_feature_extractor = nn.Sequential(*list(self.resnet.children())[:-1])  
    
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        
        self.fc_layers = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3), 

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.7),

            nn.Linear(128, num_classes)  
        )

    def forward(self, x):
        batch_size, seq_len, c, h, w = x.shape
        x = x.view(batch_size * seq_len, c, h, w)  
        features = self.resnet_feature_extractor(x) 
        features = features.view(batch_size, seq_len, -1) 

        lstm_out, _ = self.lstm(features) 
        x = lstm_out[:, -1, :]  
    
        output = self.fc_layers(x) 
        
        return output
