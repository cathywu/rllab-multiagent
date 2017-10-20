from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.algos.batch_polopt import BatchPolopt
from rllab.algos.npo import NPO


class MultiNPO(BatchPolopt):
    """
    Multi Natural Policy Optimization.
    """

    def __init__(
            self,
            env,
            optimizer,
            baselines,
            **kwargs):
        # FIXME(cathywu) if env is not passed by reference, this may cause issues
        self.NPOs = [NPO(env=shadow_env, idx=idx, optimizer=optimizer[idx],
                         baseline=baselines[idx], **kwargs) for idx, shadow_env in
                     enumerate(env.shadow_envs)]
        self.baselines = baselines
        super(MultiNPO, self).__init__(env=env, baseline=None, **kwargs)

    @overrides
    def init_opt(self):
        for optimizer in self.NPOs:
            optimizer.init_opt()
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data_n):
        for samples_data, optimizer in zip(samples_data_n, self.NPOs):
            optimizer.optimize_policy(itr, samples_data)
        return dict()

    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baselines,
            env=self.env,
        )
