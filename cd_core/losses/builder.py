import torch
from cd_core.utils import Registry,build_from_cfg

LOSSES = Registry('loss')

def build_loss(cfg, default_args=None):
    """Build loss."""
    if LOSSES.get(cfg['type']):
        loss = build_from_cfg(cfg, LOSSES, default_args=default_args)
    else:
        loss = build_from_cfg(cfg, torch.nn, default_args=default_args, mode='module')
    return loss