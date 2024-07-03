import os
import pandas as pd
import torch

from dataloaders.utils import image_loader, get_unk_mask_indices


class MultilabelDataset(torch.utils.data.Dataset):
    def __init__(self, ann_dir,root_dir, num_labels, transform=None,known_labels=0, testing=False, da_root_dir = None, itri = False):
        # Load training data.
        self.ann_dir = pd.read_csv(ann_dir)
        self.root_dir = root_dir
        self.transform = transform

        self.num_labels = num_labels
        self.known_labels = known_labels
        self.testing = testing
        
        self.da_root_dir = da_root_dir
        self.itri = itri

    def __getitem__(self, index):
        # sample = self.annData[index]
        # img_path = sample['file_path']
        # image_id = sample['image_id']

        # img_path = os.path.join(self.root_dir, str(self.ann_dir.iloc[index, 0])+".png")
        if self.itri:
            img_path = os.path.join(self.da_root_dir, str(self.ann_dir.iloc[index, 0]))
        else:
            if str(self.ann_dir.iloc[index, 0]).startswith("DA"):
                base_path = os.path.join(self.da_root_dir, str(self.ann_dir.iloc[index, 0]))
            else:
                base_path = os.path.join(self.root_dir, str(self.ann_dir.iloc[index, 0]))
            # img_path = base_path + ".png" if os.path.exists(base_path + ".png") else base_path + ".tif"
            if os.path.exists(base_path + ".png"):
                img_path = base_path + ".png"
            elif os.path.exists(base_path + ".tif"):
                img_path = base_path + ".tif"
            elif os.path.exists(base_path + ".jpg"):
                img_path = base_path + ".jpg"

        image = image_loader(img_path,self.transform)
        
        labels = torch.tensor(self.ann_dir.iloc[index, 1:].astype(float).values, dtype=torch.float32)

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