import torchvision.models as models
import torch.nn as nn

class EfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3, self).__init__()
        efficientnetb3 = models.efficientnet_b3(weights='DEFAULT')
        # efficientnetb3.classifier.in_features # 1536
        efficientnetb3.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1536, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )
        self.model = efficientnetb3

    def forward(self, x):
        return self.model(x)
