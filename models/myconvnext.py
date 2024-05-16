import torch
import torch.nn as nn
import torchvision.models as models
import timm
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from models.utils import weights_init

class ConvNeXtTransformer(nn.Module):
    def __init__(self, num_classes, nhead=8, dim_feedforward=4096, num_transformer_layers=1):
        super(ConvNeXtTransformer, self).__init__()
        self.features = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=0)
        self.flatten = nn.Flatten(start_dim=2, end_dim=-1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        d_model = 1536
        self.layer_norm = nn.LayerNorm(d_model)
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_transformer_layers, norm=nn.LayerNorm(d_model), enable_nested_tensor=False)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.features.forward_features(x)
        # x_gap = self.gap(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = torch.mean(x,dim=2)
        # x = torch.cat((x_gap.view(x_gap.size(0), -1), x), dim=1)
        x = self.head(x)
        return x
    
    
class myConvNeXt1(nn.Module):
    def __init__(self, num_classes):
        super(myConvNeXt1, self).__init__()
        self.features = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, features_only=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        # self.classifier = nn.Sequential(
        #     nn.Linear(2880, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes),
        # )
        self.classifier = nn.Linear(2880, num_classes)
        self.classifier.apply(weights_init)
        
    def forward(self, x):
        features = []
        output = self.features(x)
        for feature in output:
            features.append(self.gap(feature))
        x = torch.cat(features, dim=1)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    

class myConvNeXt2(nn.Module):
    def __init__(self, num_classes):
        super(myConvNeXt2, self).__init__()
        self.convnext = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True)
        self.convnext.head = nn.Identity()
        self.conv1x1 = nn.Conv2d(1536, num_classes, kernel_size=1)
        self.bn1 = nn.BatchNorm2d(num_features=1536)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.relu = nn.ReLU()
        
        self.conv1x1.apply(weights_init)
        
    def forward(self, x):
        x = self.convnext(x)
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1x1(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        return x


class myConvNeXt3(nn.Module):
    def __init__(self, num_classes):
        super(myConvNeXt3, self).__init__()
        self.convnext = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True)
        self.convnext.head = nn.Identity()
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=1536, num_heads=1, dropout=0.1, batch_first=True)
        self.layer_norm = nn.LayerNorm(1536)
        # self.classifier = nn.Sequential(
        #     nn.Linear(1536, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes),
        # )
        self.classifier = nn.Linear(1536, num_classes)
        self.classifier.apply(weights_init)
        
    def forward(self, x):
        x = self.convnext(x)
        x = self.to_features(x)
        x = x.permute(0, 2, 1)
        # x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        x = x.permute(0, 2, 1)
        # print(x.shape)
        x = torch.mean(x, dim=2)
        # print(x.shape)
        x = self.classifier(x)
        return x
    
    
class myConvNeXt4(nn.Module):
    def __init__(self, num_classes):
        super(myConvNeXt4, self).__init__()
        self.convnext = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True)
        self.convnext.head = nn.Identity()
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=1536, num_heads=1, dropout=0.1, batch_first=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_norm = nn.LayerNorm(1536)
        # self.classifier = nn.Sequential(
        #     nn.Linear(1536, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes),
        # )
        self.classifier = nn.Linear(1536, num_classes)
        
        self.classifier.apply(weights_init)
        
    def forward(self, x):
        x = self.convnext(x)
        cls_tokens = self.gap(x)
        cls_tokens = cls_tokens.squeeze().unsqueeze(1)
        x = self.to_features(x)
        x = x.permute(0, 2, 1)
        x = torch.cat((cls_tokens, x), dim=1)
        # x = self.layer_norm(x)
        x, _ = self.attention(x, x, x)
        x = self.classifier(x[:, 0, :])
        return x
    
    
class myConvNeXt5(nn.Module):
    def __init__(self, num_classes):
        super(myConvNeXt5, self).__init__()
        self.convnext = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True)
        self.convnext.head = nn.Identity()
        self.to_features = nn.Flatten(start_dim=2)
        self.attention = nn.MultiheadAttention(embed_dim=1536, num_heads=1, dropout=0.1, batch_first=True)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.layer_norm = nn.LayerNorm(1536)
        # self.classifier = nn.Sequential(
        #     nn.Linear(1536, 512),
        #     nn.BatchNorm1d(512),
        #     nn.ReLU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(512, num_classes),
        # )
        self.classifier = nn.Linear(1536, num_classes)
        
        self.classifier.apply(weights_init)
        
    def forward(self, x):
        x = self.convnext(x)
        
        return x