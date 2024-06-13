import torch.nn as nn
import torchvision.models as models
import timm
from models.utils import weights_init

class MaxViT(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.maxvit = timm.create_model('maxvit_base_tf_384.in21k_ft_in1k', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.maxvit(x)