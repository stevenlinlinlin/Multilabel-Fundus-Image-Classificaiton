import torch.nn as nn
import torchvision.models as models
from models.utils import weights_init

class SwinTransformer(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.swin = models.swin_v2_b(weights='DEFAULT')
        self.swin.head = nn.Linear(self.swin.head.in_features, num_classes)
        self.swin.head.apply(weights_init)

    def forward(self, x):
        return self.swin(x)