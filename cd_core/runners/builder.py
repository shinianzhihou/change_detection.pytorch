import torch
from cd_core.utils import Registry,build_from_cfg

HOOKS = Registry('hook')
RUNNERS = Registry('runner')
OPTIMIZERS = Registry('optimizer')
LR_SCHEDULERS = Registry('lr_scheduler')

def build_optimizer(cfg, default_args=None):
    """Build optimizer."""
    if OPTIMIZERS.get(cfg['type']):
        optimizer = build_from_cfg(cfg, OPTIMIZERS, default_args=default_args)
    else:
        optimizer = build_from_cfg(cfg, torch.optim, default_args=default_args, mode='module')
    return optimizer
    
def build_lr_scheduler(cfg, default_args=None):
    if LR_SCHEDULERS.get(cfg['type']):
        lr_scheduler = build_from_cfg(cfg, LR_SCHEDULERS, default_args=default_args)
    else:
        lr_scheduler = build_from_cfg(cfg, torch.optim.lr_scheduler, default_args=default_args, mode='module')
    return lr_scheduler