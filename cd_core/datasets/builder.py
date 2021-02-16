import albumentations as A

from cd_core.utils import Registry,build_from_cfg


DATASETS = Registry('dataset')
PIPELINES = Registry('pipeline')



def build_dataset(cfg, default_args=None):
    """Build dataset."""
    dataset = build_from_cfg(cfg, DATASETS, default_args)
    return dataset

# TODO(snian) Separate `Compose` from `build_pipeline`.
def build_pipeline(cfgs, default_args=None):
    """Build pipeline."""
    assert isinstance(cfgs,list), \
        f'Configs for pipeline must be a list, but get {type(cfgs)}.'
    transformers = []
    for cfg in cfgs:
        if PIPELINES.get(cfg['type']):
            transformer = build_from_cfg(cfg, PIPELINES)
        else:
            transformer = build_from_cfg(cfg, A, mode='module')
        transformers.append(transformer)
    pipeline = A.Compose(transformers, additional_targets={'image1': 'image'})
    return pipeline