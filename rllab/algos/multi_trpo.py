
from rllab.algos.multi_npo import MultiNPO
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class MultiTRPO(MultiNPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        n=len(kwargs['policy'].policies)
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizers = [ConjugateGradientOptimizer(**optimizer_args) for _ in range(n)]
        super(MultiTRPO, self).__init__(optimizer=optimizers, **kwargs)
