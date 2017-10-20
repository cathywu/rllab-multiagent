from rllab.core.lasagne_powered import LasagnePowered
import lasagne.layers as L
from rllab.core.network import MLP
from rllab.core.serializable import Serializable
from rllab.policies.base import StochasticPolicy
from rllab.misc import ext
from sandbox.rocky.tf.misc import tensor_utils
from rllab.misc.overrides import overrides
#from sandbox.rocky.tf.regressors.auto_mlp_regressor import space_to_dist_dim, output_to_info, space_to_distribution
#from sandbox.rocky.tf.spaces.discrete import Discrete
from rllab.spaces.box import Box


class MultiMLPPolicy(StochasticPolicy, LasagnePowered, Serializable):
    def __init__(
            self,
            name,
            env_spec,
            policies,
            mean_network=None,

    ):
        """
        :param name: base tf variable name
        :param env_spec: A spec for the mdp.
        :param policies: list of policies
        :return:
        """
        Serializable.quick_init(self, locals())

        self.policies = policies
        self.n = len(policies)
        self.training = True  # training vs testing
        self.env_spec = env_spec
        super(MultiMLPPolicy, self).__init__(env_spec)
        LasagnePowered.__init__(self, [self.policies[0]._l_mean, self.policies[0]._l_log_std])

    @property
    def vectorized(self):
        return True

    @overrides
    def dist_info_sym(self, obs_var, state_info_vars=None):
        return NotImplementedError

    @overrides
    def dist_info(self, obs, state_infos=None):
        return NotImplementedError

    # The return value is a pair. The first item is a matrix (N, A), where each
    # entry corresponds to the action value taken. The second item is a vector
    # of length N, where each entry is the density value for that action, under
    # the current policy
    @overrides
    def get_action(self, multi_observation):
        # TODO(cathywu) Note: this doesn't seem to be called
        # TODO(cathywu) change to handle multi observations
        actions, agent_infos = self.get_actions([multi_observation])
        return actions[0], {k: v[0] for k, v in agent_infos[0].items()}

    def get_actions(self, observations, env_infos=None):
        """
        Get actions for n agents from a list of n observations

        :param observations: [[[obs]*n_agents]*n_vec_envs]
        :param env_infos:
        :return: actions [([actions]*n_agents)*n_vec_envs]
        :return: dist_info { key: n_vec_envs x sum_n_agents_dims }
        """
        # Note: structure is observations for 40 vec_envs
        # TODO(cathywu) separate out physical observations from virtual; how to acquire state?
        # TODO(cathywu) change to handle multi observations
        import numpy as np
        observations_n = list(zip(*observations))
        n = len(observations_n)
        if self.training and env_infos is not None:
            # Note: in the first step of training, env_infos == None, so the
            # observations are used
            states = [x['state'] for x in env_infos]
            # TODO(cathywu) DURING TRAINING: Ignore observations and use o_hat instead
            # FIXME(cathywu) This is in lieu of the o_hat function for now
            observations_n = list(zip(*states))
            flat_obs_n = [
                self.env_spec[i].observation_space.flatten_n(observations_n[i])
                for i in range(n)]
            # observations = self.observation_approx.map(observations)
        else:
            flat_obs_n = [
                self.env_spec[i].observation_space.flatten_n(observations_n[i])
                for i in range(n)]
        # FIXME(eugene) check if log_std is being computed correctly...
        dist_info_n = [dict(zip(['mean', 'log_std'], self.policies[i]._f_dist(flat_obs_n[i]))) for i in
                       range(n)]
        actions_n = [
            [[x] for x in self.get_distribution(idx=i).sample(dist_info_n[i])]
            for i in range(n)]
        # TODO(cathywu) check that this works for more than 2 agents
        actions = [x for x in zip(*actions_n)]
        return actions, dist_info_n

    @property
    def distribution(self):
        return NotImplementedError

    def get_distribution(self, idx=0):
        return self.policies[idx]._dist
