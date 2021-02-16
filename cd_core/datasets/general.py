import os
import pandas as pd

from cd_core.datasets import BaseDataset

from .builder import DATASETS


@DATASETS.register_module()
class General(BaseDataset):
    """A general dataset class.

    meta_file:
        image_000_a.jpg image_000_b.jpg gt_000.png
        image_001_a.jpg image_001_b.jpg gt_001.png
        image_002_a.jpg image_002_b.jpg gt_002.png
        ...
    
    Args:
        meta_file (str): Path to meta_file.
        pipeline (list[dict]): Processing pipeline.
        data_root (str): Data root for images.
        test_mode (bool): If test_mode=True, gt wouldn't be loaded.

    """
    def __init__(self,
                 meta_file,
                 data_root='',
                 pipeline=None,
                 test_mode=False,
                ):
        super().__init__()
        self.data_root = ''
        self.pipeline = pipeline
        self.test_mode = test_mode
        self.data = pd.read_csv(meta_file,header=None)
        self.length = len(self.data)


    def __getitem__(self,idx):

        return


    def __len__(self):
        return self.length

    
