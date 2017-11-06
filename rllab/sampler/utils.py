import numpy as np
from rllab.misc import tensor_utils
import time


def rollout(env, agent, max_path_length=np.inf, animated=False, speedup=1):
    observations = []
    actions = []
    rewards = []
    agent_infos = []
    env_infos = []
    dones = []
    o = env.reset()
    agent.reset()
    path_length = 0
    if animated:
        env.render()
    while path_length < max_path_length:

        # if we have a multirnn policy, it will need to output multiple actions for a single step
        if hasattr(env, 'num_actions'):
            num_actions = env.num_actions
        else:
            num_actions = 1
        d = 0
        for _ in range(num_actions):
            a, agent_info = agent.get_action(o)
            next_o, r, d, env_info = env.step(a)
            if isinstance(env.observation_space, list):
                n = len(env.shadow_envs)
                observations.append(
                    [env.shadow_envs[i].observation_space.flatten_n(o[i]) for i in
                     range(n)])
                rewards.append(r)
                actions.append(
                    [env.shadow_envs[i].action_space.flatten_n(a[i]) for i in range(n)])
            else:
                observations.append(env.observation_space.flatten(o))
                rewards.append(r)
                actions.append(env.action_space.flatten(a))
            agent_infos.append(agent_info)
            env_infos.append(env_info)
            dones.append(d)
            path_length += 1
            if d:
                break
            o = next_o
        if d:
            break
        if animated:
            env.render()
            timestep = 0.05
            time.sleep(timestep / speedup)

    # this is a gross hack for now
    if animated:
        env.render(close=True)

    return dict(
        observations=tensor_utils.stack_tensor_list(observations),
        actions=tensor_utils.stack_tensor_list(actions),
        rewards=tensor_utils.stack_tensor_list(rewards),
        agent_infos=tensor_utils.stack_tensor_dict_list(agent_infos),
        env_infos=tensor_utils.stack_tensor_dict_list(env_infos),
        dones=np.asarray(dones),
        last_obs=o,
    )
