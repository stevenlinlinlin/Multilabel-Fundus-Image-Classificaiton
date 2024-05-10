import torch
import torch.nn as nn
import torchvision.models as models
from models.utils import weights_init
from models.transformerencoder import SelfAttnLayer

class CustomDenseNet1(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet1, self).__init__()
        self.features = models.densenet161(weights='DEFAULT').features
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Linear(5472, num_classes)
        self.classifier = nn.Sequential(
            # nn.BatchNorm1d(num_features),
            # nn.LayerNorm(2208),
            nn.Linear(5472, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )
        
        self.classifier.apply(weights_init)
        
    def forward(self, x):
        features = []
        for name, module in self.features.named_children():
            x = module(x)
            if 'denseblock' in name:
                # print(self.gap(x).size())
                features.append(self.gap(x))
        x = torch.cat(features, dim=1)
        # x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class CustomDenseNet2(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet2, self).__init__()
        self.features = models.densenet161(weights='DEFAULT').features
        self.conv = nn.Conv2d(2208, 28, kernel_size=1)
        self.bn = nn.BatchNorm2d(num_features=2208)
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=2208, num_heads=1, dropout=0.1, batch_first=True)
        self.layer_norm = nn.LayerNorm(2208)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
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
        # self.relu = nn.ReLU()
        # self.classifier = nn.Linear(49, num_classes)
        
        self.classifier.apply(weights_init)
        
    def forward(self, x):
        x = self.features(x)
        # x = self.bn(x)
        # x_gap= self.gap(x)
        x = self.to_features(x)
        x = x.permute(0, 2, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1)
        x = torch.mean(x, dim=2)
        # x = torch.cat([x,x_gap.view(x_gap.size(0), -1)], dim=1)
        # x = x + x_gap.view(x_gap.size(0), -1)
        # x = self.dropout(x)
        x = self.classifier(x)
        return x
    
    
class CustomDenseNet3(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet3, self).__init__()
        self.features = models.densenet161(weights='DEFAULT').features
        self.bn = nn.BatchNorm2d(num_features=2208)
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=2208, num_heads=1, dropout=0.1, batch_first=True)
        # self.cls_token = nn.Parameter(torch.randn(1, 1, 2208))
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
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
        # self.classifier = nn.Linear(2208, num_classes)
        
        self.classifier.apply(weights_init)
        
    def forward(self, x):
        x = self.features(x)
        # x = self.bn(x)
        
        cls_tokens = self.gap(x)
        cls_tokens = cls_tokens.squeeze().unsqueeze(1)
        # print(cls_tokens.size())
        
        x = self.to_features(x)
        x = x.permute(0, 2, 1)
        
        # batch_size = x.size(0)
        # cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        # print(x.size())
        
        x, _ = self.attention(x, x, x)
        # gap_tokens = x[:, 0, :]
        # x = torch.mean(x[:,1:,:].permute(0, 2, 1), dim=2)
        # x = self.relu(x)
        # x = cls_tokens.view(cls_tokens.size(0),-1) + x[:, 0, :]
        # x = self.layer_norm(x)
        # x = torch.cat([x, gap_tokens, cls_tokens.squeeze()], dim=1)
        # x = x + gap_tokens
        x = self.classifier(x[:, 0, :])
        return x
    

class CustomDenseNet4(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet4, self).__init__()
        self.features = models.densenet161(weights='DEFAULT').features
        self.conv1x1 = nn.Conv2d(2208, num_classes, kernel_size=1)
        self.conv1 = nn.Conv2d(2208, 552, kernel_size=1)
        self.conv2 = nn.Conv2d(552, num_classes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=2208)
        self.bn2 = nn.BatchNorm2d(num_features=num_classes)
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=num_classes, num_heads=1, dropout=0.1, batch_first=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(num_classes)
        self.classifier = nn.Linear(2208, num_classes)
        
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv1x1.apply(weights_init)
        
    def forward(self, x):
        x = self.features(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        x = self.conv1x1(x)
        # x = self.conv1(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.conv2(x)
        # x = self.bn2(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x
    
    
class CustomDenseNet5(nn.Module):
    def __init__(self, num_classes):
        super(CustomDenseNet5, self).__init__()
        self.features = models.densenet161(weights='DEFAULT').features
        self.conv1x1 = nn.Conv2d(2208, num_classes, kernel_size=1)
        self.conv1 = nn.Conv2d(2208, 552, kernel_size=1)
        self.conv2 = nn.Conv2d(552, num_classes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=2208)
        self.bn2 = nn.BatchNorm2d(num_features=num_classes)
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=49, num_heads=1, dropout=0.1, batch_first=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.layer_norm = nn.LayerNorm(num_classes)
        self.classifier = nn.Linear(49, 1)
        
        self.conv1.apply(weights_init)
        self.conv2.apply(weights_init)
        self.conv1x1.apply(weights_init)
        
    def forward(self, x):
        x = self.features(x)
        # x = self.bn1(x)
        # x = self.relu(x)
        # print(x.shape)
        x = self.conv1x1(x)
        # print(x.shape)
        x = self.to_features(x)
        x,_ = self.attention(x,x,x)
        x = self.classifier(x)
        print(x.shape)
        x = x.view(x.size(0), -1)
        return x