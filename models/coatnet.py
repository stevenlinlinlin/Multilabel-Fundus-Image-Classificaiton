import torch.nn as nn
import torchvision.models as models
import timm
from models.utils import weights_init

class CoAtNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.coatnet = timm.create_model('coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.coatnet(x)