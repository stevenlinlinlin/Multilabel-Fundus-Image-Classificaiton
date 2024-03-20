import os
from PIL import Image
import pandas as pd
import torch
  
class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, str(self.annotations.iloc[index, 0])+".png")
        image = Image.open(img_path).convert("RGB")
        labels = torch.tensor(self.annotations.iloc[index, 1:].values, dtype=torch.float32)
        
        if self.transform:
            image = self.transform(image)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        return sample