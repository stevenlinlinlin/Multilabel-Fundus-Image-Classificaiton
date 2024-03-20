import torchvision.models as models
import torch.nn as nn
from transformers import ViTForImageClassification, AutoImageProcessor


class ViTForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels):
        super(ViTForMultiLabelClassification, self).__init__()
        self.processor = AutoImageProcessor.from_pretrained('google/vit-base-patch16-224-in21k')
        self.vit = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k', num_labels=num_labels)
        self.vit.classifier = nn.Sequential(
            nn.Linear(self.vit.classifier.in_features, num_labels),
        )

    def forward(self, image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.vit(inputs)
        return outputs.logits
   