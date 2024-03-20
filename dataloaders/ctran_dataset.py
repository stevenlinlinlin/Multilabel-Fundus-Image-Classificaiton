import os
from PIL import Image
import pandas as pd
import numpy as np
import random
import time
import hashlib
import torch

def get_unk_mask_indices(image,testing,num_labels,known_labels):
    if testing:
        # for consistency across epochs and experiments, seed using hashed image array 
        random.seed(hashlib.sha1(np.array(image)).hexdigest())
        unk_mask_indices = random.sample(range(num_labels), (num_labels-int(known_labels)))
    else:
        # sample random number of known labels during training
        if known_labels>0:
            random.seed()
            num_known = random.randint(0,int(num_labels*0.75))
        else:
            num_known = 0

        unk_mask_indices = random.sample(range(num_labels), (num_labels-num_known))

    return unk_mask_indices

def image_loader(path,transform):
    try:
        image = Image.open(path)
    except FileNotFoundError: # weird issues with loading images on our servers
        # print('FILE NOT FOUND')
        time.sleep(10)
        image = Image.open(path)

    image = image.convert('RGB')

    if transform is not None:
        image = transform(image)

    return image

class CTranDataset(torch.utils.data.Dataset):
    def __init__(self, ann_dir,root_dir,transform=None,known_labels=0, testing=False):
        # Load training data.
        self.ann_dir = pd.read_csv(ann_dir)
        self.root_dir = root_dir
        self.transform = transform

        self.num_labels = 28
        self.known_labels = known_labels
        self.testing = testing

    def __getitem__(self, index):
        # sample = self.annData[index]
        # img_path = sample['file_path']
        # image_id = sample['image_id']

        img_path = os.path.join(self.root_dir, str(self.ann_dir.iloc[index, 0])+".png")

        image = image_loader(img_path,self.transform)
        
        labels = torch.tensor(self.ann_dir.iloc[index, 1:].values, dtype=torch.float32)

        mask = labels.clone()
        unk_mask_indices = get_unk_mask_indices(image,self.testing,self.num_labels,self.known_labels)
        mask.scatter_(0,torch.Tensor(unk_mask_indices).long() , -1)

        sample = {}
        sample['image'] = image
        sample['labels'] = labels
        sample['mask'] = mask
        # sample['imageIDs'] = str(image_id)

        return sample

    def __len__(self):
        return len(self.ann_dir)