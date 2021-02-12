
from cd_core.utils import Registry,build_from_cfg

DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')

def build_dataset(cfg, default_args=None):
    """Build datasets."""
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset