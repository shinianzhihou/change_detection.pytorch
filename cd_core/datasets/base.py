from torch.utils.data import Dataset
from .builder import DATASETS

@DATASETS.register_module()
class BaseDataset(Dataset):

    CLASSES = ["unchanged","changed"]

    PALETTE = [[0,0,0],[255,255,255]]

    def __init__(self, pipeline=None):
        self.pipeline = pipeline

    def process(self, images, mask):
        # TODO(snian) Also make the number of masks configurable
        if self.pipeline:
            augmented = self.pipeline(image=images['image0'],image1=images['image1'], mask=mask)
            images['image0'] = augmented['image']
            images['image1'] = augmented['image1']
            images['mask']

        return images, mask
