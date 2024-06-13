import torch.nn as nn
import torchvision.models as models
import timm
from models.utils import weights_init

class MViT_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.mvit_v2 = timm.create_model('mvitv2_base.fb_in1k', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.mvit_v2(x)