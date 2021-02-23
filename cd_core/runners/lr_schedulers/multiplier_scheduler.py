from typing import List

import torch
from cd_core.runners.lr_schedulers import ParamScheduler

class MultiplierScheduler(torch.optim.lr_scheduler._LRScheduler):

    def __init__(
            self,
            optimizer: torch.optim.Optimizer,
            multiplier: ParamScheduler,
            max_iter: int,
            last_iter: int = -1,
        ):
            """
            Args:
                optimizer, last_iter: See ``torch.optim.lr_scheduler._LRScheduler``.
                    ``last_iter`` is the same as ``last_epoch``.
                multiplier: a fvcore ParamScheduler that defines the multiplier on
                    every LR of the optimizer
                max_iter: the total number of training iterations
            """
            if not isinstance(multiplier, ParamScheduler):
                raise ValueError(
                    "_LRMultiplier(multiplier=) must be an instance of fvcore "
                    f"ParamScheduler. Got {multiplier} instead."
                )
            self._multiplier = multiplier
            self._max_iter = max_iter
            super().__init__(optimizer, last_epoch=last_iter)

    def state_dict(self):
        # fvcore schedulers are stateless. Only keep pytorch scheduler states
        return {"base_lrs": self.base_lrs, "last_epoch": self.last_epoch}

    def get_lr(self) -> List[float]:
        multiplier = self._multiplier(self.last_epoch / self._max_iter)
        return [base_lr * multiplier for base_lr in self.base_lrs]
