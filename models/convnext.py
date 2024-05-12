import torch.nn as nn
import torchvision.models as models
import timm
from models.utils import weights_init

class ConvNeXt(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
         # self.convnext = models.convnext_large(weights='DEFAULT')
        self.convnext = timm.create_model('convnextv2_large.fcmae_ft_in22k_in1k_384', pretrained=True, num_classes=num_classes)
        # self.convnext.classifier = nn.Sequential(
        #     nn.LayerNorm([1536,1,1], elementwise_affine=True),
        #     nn.Flatten(start_dim=1,end_dim=-1),
        #     nn.Linear(1536, num_classes)
        # )
        # self.convnext.classifier.apply(weights_init)

    def forward(self, x):
        return self.convnext(x)