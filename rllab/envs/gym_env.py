import gym
import gym.wrappers
import gym.envs
import gym.spaces
import traceback
import logging
from gym.envs.registration import register

import os
import os.path as osp
from rllab.envs.base import Env, Step
from rllab.core.serializable import Serializable
from rllab.spaces.box import Box
from rllab.spaces.discrete import Discrete
from rllab.spaces.product import Product
from rllab.misc import logger
from rllab.envs.shadow_env import ShadowEnv

env_version_num = 0


def convert_gym_space(space):
    if isinstance(space, gym.spaces.Box):
        return Box(low=space.low, high=space.high)
    elif isinstance(space, gym.spaces.Discrete):
        return Discrete(n=space.n)
    elif isinstance(space, gym.spaces.Tuple):
        return Product([convert_gym_space(x) for x in space.spaces])
    elif isinstance(space, list):
        # For multiagent envs
        return list(map(convert_gym_space, space))
        # TODO(cathywu) refactor multiagent envs to use gym.spaces.Tuple
        # (may be needed for pickling?)
    else:
        raise NotImplementedError


class CappedCubicVideoSchedule(object):
    # Copied from gym, since this method is frequently moved around
    def __call__(self, count):
        if count < 1000:
            return int(round(count ** (1. / 3))) ** 3 == count
        else:
            return count % 1000 == 0


class FixedIntervalVideoSchedule(object):
    def __init__(self, interval):
        self.interval = interval

    def __call__(self, count):
        return count % self.interval == 0


class NoVideoSchedule(object):
    def __call__(self, count):
        return False


class GymEnv(Env, Serializable):
    def __init__(self, env_name, record_video=True, video_schedule=None, log_dir=None, record_log=True,
                 force_reset=False, register_params=None):
        if log_dir is None:
            if logger.get_snapshot_dir() is None:
                logger.log("Warning: skipping Gym environment monitoring since snapshot_dir not configured.")
            else:
                log_dir = os.path.join(logger.get_snapshot_dir(), "gym_log")
        Serializable.quick_init(self, locals())

        pass_params(*register_params)
        env = gym.envs.make(env_name+'-v'+str(env_version_num))
        self.env = env
        self.env_id = env.spec.id

        assert not (not record_log and record_video)

        if log_dir is None or record_log is False:
            self.monitoring = False
        else:
            if not record_video:
                video_schedule = NoVideoSchedule()
            else:
                if video_schedule is None:
                    video_schedule = CappedCubicVideoSchedule()
            self.env = gym.wrappers.Monitor(self.env, log_dir, video_callable=video_schedule, force=True)
            self.monitoring = True

        self._observation_space = convert_gym_space(env.observation_space)
        logger.log("observation space: {}".format(self._observation_space))
        self._action_space = convert_gym_space(env.action_space)
        logger.log("action space: {}".format(self._action_space))
        self._horizon = env.spec.tags['wrapper_config.TimeLimit.max_episode_steps']
        self._log_dir = log_dir
        self._force_reset = force_reset
        # For multi-agent envs
        self._shadow_envs = self.construct_shadow_envs()

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def action_space(self):
        return self._action_space

    @property
    def horizon(self):
        return self._horizon

    def reset(self):
        if self._force_reset and self.monitoring:
            from gym.wrappers.monitoring import Monitor
            assert isinstance(self.env, Monitor)
            recorder = self.env.stats_recorder
            if recorder is not None:
                recorder.done = True
        return self.env.reset()

    def step(self, action):
        next_obs, reward, done, info = self.env.step(action)
        return Step(next_obs, reward, done, **info)

    def render(self, close=None):
        self.env.render()

    def terminate(self):
        if self.monitoring:
            self.env._close()
            if self._log_dir is not None:
                print("""
    ***************************

    Training finished! You can upload results to OpenAI Gym by running the following command:

    python scripts/submit_gym.py %s

    ***************************
                """ % self._log_dir)

    @property
    def shadow_envs(self):
        """
        For multi-agent envs
        :return:
        """
        return self._shadow_envs

    def construct_shadow_envs(self):
        """
        For multi-agent envs: Construct shadow envs for each of the agents.
        :return:
        """
        shadow_envs = []
        if not isinstance(self.observation_space, list):
            return shadow_envs
        for obs, action in zip(self.observation_space, self.action_space):
            shadow_envs.append(ShadowEnv(observation_space=obs, action_space=action))
        return shadow_envs


def pass_params(env_name, sumo_params, type_params, env_params, net_params,
                initial_config, scenario):
    global env_version_num

    env_version_num += 1
    register(
        id=env_name+'-v'+str(env_version_num),
        entry_point='flow.envs:'+env_name,
        max_episode_steps=env_params.horizon,
        kwargs={"env_params": env_params, "sumo_params": sumo_params, "scenario": scenario}
    )
