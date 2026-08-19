from torch.utils.data import Dataset
import skimage as ski
import numpy as np
import albumentations as A
import torch
import cv2

class SegDataset(Dataset):
    def __init__(self, dataset, transform=None):
        super(SegDataset, self).__init__()
        self.dataset = dataset.reset_index()
        self.transform = transform

    def __str__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        
        image = cv2.imread(self.dataset.loc[idx, "image"])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        mask = cv2.imread(self.dataset.loc[idx, "mask"], cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask >= 127, 1, 0).astype(np.uint8)  

        
        if self.transform is not None:
            transformed = self.transform(image=image, mask=mask)
            image = transformed["image"]
            mask = transformed["mask"]
        mask = mask[torch.newaxis, ...]
        return image.float(), mask.float()
    
    def __len__(self):
        return len(self.dataset)
    
    def get_minimum(self):
        minimum = np.inf
        for i, _ in self.dataset.iterrows():
            im = ski.io.imread(self.dataset.loc[i, "im"])
            min_shape = min(im.shape[0], im.shape[1])
            minimum = min(minimum, min_shape)
        print("minimum width and height:", minimum)