

from rllab.algos.shared_npo import SharedNPO
from rllab.optimizers.conjugate_gradient_optimizer import ConjugateGradientOptimizer


class SharedTRPO(SharedNPO):
    """
    Trust Region Policy Optimization
    """

    def __init__(
            self,
            optimizer=None,
            optimizer_args=None,
            **kwargs):
        if optimizer is None:
            if optimizer_args is None:
                optimizer_args = dict()
            optimizer = ConjugateGradientOptimizer(**optimizer_args)
        super(SharedTRPO, self).__init__(optimizer=optimizer, **kwargs)
