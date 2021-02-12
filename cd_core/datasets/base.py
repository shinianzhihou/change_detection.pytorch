from torch.utils.data import Dataset


class BaseDataset(Dataset):

    CLASSES = ["unchanged","changed"]

    PALETTE = [(0,0,0),(255,255,255)]

    def __init__(self, transform=None):
        self.transform = transform

    def process(self, image, masks):
        if self.transform:
            augmented = self.transform(image=image, masks=masks)
            return augmented['image'], augmented['masks']
        else:
            return image, masks
