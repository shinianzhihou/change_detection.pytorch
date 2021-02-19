import torch
from cd_core.utils import Registry,build_from_cfg

OPTIMIZERS = Registry('optimizer')

def build_optimizer(cfg, default_args=None):
    """Build optimizer."""
    if OPTIMIZERS.get(cfg['type']):
        optimizer = build_from_cfg(cfg, OPTIMIZERS, default_args=default_args)
    else:
        optimizer = build_from_cfg(cfg, torch.optim, default_args=default_args, mode='module')
    return optimizer