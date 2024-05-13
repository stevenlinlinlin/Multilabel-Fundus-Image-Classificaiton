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
        d_model = 1536
        encoder_layers = TransformerEncoderLayer(d_model, nhead, dim_feedforward, batch_first=True)
        self.transformer_encoder = TransformerEncoder(encoder_layers, num_transformer_layers)
        self.layer_norm = nn.LayerNorm(1536)
        self.head = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        x = self.features.forward_features(x)
        x = self.flatten(x)
        x = x.permute(0, 2, 1)
        x = self.layer_norm(x)
        x = self.transformer_encoder(x)
        x = x.permute(0, 2, 1)
        x = torch.mean(x,dim=2)
        x = self.head(x)
        return x