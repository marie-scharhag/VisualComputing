import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class PlantDataset(Dataset):
    def __init__(self,image_list,labels_list,transform=None):
        self.images = image_list
        self.labels = labels_list
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):
        image = np.array(Image.open(self.images[item]).convert('RGB'))
        label = np.array(Image.open(self.labels[item]))

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=label)
            image = augmentations['image']
            label = augmentations['mask']

        return image, label