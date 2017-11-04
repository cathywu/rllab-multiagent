from rllab.misc import ext
from rllab.misc.overrides import overrides
import rllab.misc.logger as logger
from rllab.optimizers.penalty_lbfgs_optimizer import PenaltyLbfgsOptimizer
from rllab.algos.batch_polopt import BatchPolopt
from rllab.algos.npo import NPO
import numpy as np

class SharedNPO(BatchPolopt):
    """
    Multi Natural Policy Optimization.
    """

    def __init__(
            self,
            env,
            optimizer,
            baselines,
            NPO_cls=NPO,
            **kwargs):
        # FIXME(cathywu) if env is not passed by reference, this may cause issues
        # TOFIX(eugene) currently assumes that all the shadow envs are the same in action and state space
        self.NPO = NPO_cls(env=env.shadow_envs[0], idx=0, optimizer=optimizer,
                         baseline=baselines, **kwargs)
        self.baseline = baselines
        super(SharedNPO, self).__init__(env=env, baseline=None, **kwargs)

    @overrides
    def init_opt(self):
        self.NPO.init_opt()
        return dict()

    @overrides
    def optimize_policy(self, itr, samples_data_n):
        concat_samples = self.concat_dict(samples_data_n)
        self.NPO.optimize_policy(itr, concat_samples)
        return dict()

    def concat_dict(self, samples_data_n):
        n = len(samples_data_n)
        observations = np.vstack([samples_data["observations"] for samples_data in samples_data_n])
        actions = np.vstack([samples_data["actions"] for samples_data in samples_data_n])
        rewards = np.hstack([samples_data["rewards"] for samples_data in samples_data_n])
        returns = np.hstack([samples_data["returns"] for samples_data in samples_data_n])
        advantages = np.hstack([samples_data["advantages"] for samples_data in samples_data_n])
        #agent_infos = samples_data_n[0]["agent_infos"]
        agent_infos = {k: np.vstack([samples_data["agent_infos"][k] for samples_data in samples_data_n])
                       for k in samples_data_n[0]["agent_infos"].keys()}
        return dict(
            observations=observations,
            actions=actions,
            rewards=rewards,
            returns=returns,
            advantages=advantages,
            agent_infos=agent_infos
        )


    @overrides
    def get_itr_snapshot(self, itr, samples_data):
        return dict(
            itr=itr,
            policy=self.policy,
            baseline=self.baseline,
            env=self.env,
        )