from cd_core.runners import Hook,HOOKS

@HOOKS.register_module()
class LrSchedulerHook(Hook):
    # def __init__(self, ):
    def get_where(self, runner):
        """Get the proportion of task that has completed."""
        where = runner.iter / runner.max_iters
        assert 0.0<=where<=1.0, \
            f'Returned `where`({runner.iter}/{runner.max_iters} must be 0~1)'
        return where


    def after_train_iter(self, runner):
        runner.scheduler.step()

