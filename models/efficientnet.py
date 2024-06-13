import torchvision.models as models
import torch.nn as nn
import timm
from models.utils import weights_init


class EfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3, self).__init__()
        efficientnetb3 = models.efficientnet_b3(weights='DEFAULT')
        efficientnetb3.classifier = nn.Linear(1536, num_classes)
        efficientnetb3.classifier.apply(weights_init)
        self.model = efficientnetb3

    def forward(self, x):
        return self.model(x)
    
class EfficientNetB5(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB5, self).__init__()
        efficientnetb5 = models.efficientnet_b5(weights='DEFAULT')
        # efficientnetb3.classifier.in_features # 2048
        efficientnetb5.classifier = nn.Linear(2048, num_classes)
        efficientnetb5.classifier.apply(weights_init)
        self.model = efficientnetb5

    def forward(self, x):
        return self.model(x)
    
    
class EfficientNetB7(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7, self).__init__()
        efficientnetb7 = models.efficientnet_b7(weights='DEFAULT')
        # efficientnetb7.classifier.in_features # 2560
        # efficientnetb7.classifier = nn.Linear(2560, num_classes)
        efficientnetb7.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(2560, num_classes)
        )
        efficientnetb7.classifier.apply(weights_init)
        self.model = efficientnetb7

    def forward(self, x):
        return self.model(x)
    

class EfficientNet_v2(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.efficientV2 = timm.create_model('tf_efficientnetv2_xl.in21k_ft_in1k', pretrained=True, num_classes=num_classes)

    def forward(self, x):
        return self.efficientV2(x)
