import torch
import torch.nn as nn
import torchvision.models as models

class CustomDenseNet1(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet1, self).__init__()
        self.features = models.densenet161(weights='DEFAULT').features
        self.reduction = nn.Conv2d(2208, num_classes, kernel_size=1, stride=1, padding=0)
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=49, num_heads=1, dropout=0.1, batch_first=True)
        # self.classifier = nn.Linear(49, 1)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(num_features=2208)
        
    def forward(self, x):
        x = self.features(x)
        x = self.reduction(x)
        # x = self.relu(x)
        x = self.to_features(x)
        x, _ = self.attention(x, x, x)
        x = self.relu(x)
        x = torch.mean(x, dim=2)
        # x = self.classifier(x)
        return x
    

class CustomDenseNet2(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet2, self).__init__()
        self.features = models.densenet161(weights='DEFAULT').features
        # self.features = models.densenet121(weights='DEFAULT').features
        self.bn = nn.BatchNorm2d(num_features=2208)
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=2208, num_heads=8, dropout=0.1, batch_first=True)
        self.layer_norm = nn.LayerNorm(2208)
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(num_features),
            # nn.LayerNorm(2208),
            nn.Linear(2208, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        self.dropout = nn.Dropout(0.5)
        # self.classifier = nn.Linear(49, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        # x = self.bn(x)
        x = self.to_features(x)
        x = x.permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        # x = x + self.dropout(x_o)
        # x = self.layer_norm(x)
        x = x.permute(0, 2, 1)
        x = torch.mean(x, dim=2)
        x = self.classifier(x)
        return x
    
    
class CustomDenseNet3(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet3, self).__init__()
        self.features = models.densenet161(weights='DEFAULT').features
        # self.features = models.densenet121(weights='DEFAULT').features
        # self.reduction = nn.Conv2d(2208, num_classes, kernel_size=1, stride=1, padding=0)
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=49, num_heads=1, dropout=0.1, batch_first=True)
        self.classifier = nn.Linear(49, num_classes)
        
    def forward(self, x):
        x = self.features(x)
        # x = self.reduction(x)
        x = self.to_features(x)
        x, _ = self.attention(x, x, x)
        # x = torch.mean(x, dim=2)
        x = self.classifier(x[:, -1, :])
        return x