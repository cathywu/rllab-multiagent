import numpy as np

from rllab.algos import util
from rllab.misc import special
from rllab.misc import logger
from sandbox.rocky.tf.misc import tensor_utils

def regroup(lists):
    return [x for x in zip(*lists)]

class MultiSampleProcessor(object):

    def __init__(self, algo):
        self.algo = algo

    def process_samples(self, itr, paths):
        baselines = []
        returns = []
        n = len(paths[0]["rewards"])
        for i in range(n):
            baselines.append([])
            returns.append([])

        if len(paths) > 0 and "vf" in paths[0]["agent_infos"]:
            all_path_baselines = [
                p["agent_infos"]["vf"].flatten() for p in paths
            ]
        else:
            if hasattr(self.algo.baseline, "predict_n"):
                raise NotImplementedError
            else:
                all_path_baselines = [[self.algo.NPOs[i].baseline.predict(path, idx=i) for i in range(n)] for path in paths]

        for idx, path in enumerate(paths):
            path["advantages"] = []
            path["returns"] = []
            for i in range(n):
                path_baselines = np.append(all_path_baselines[idx][i], 0)
                deltas = path["rewards"][i] + \
                         self.algo.discount * path_baselines[1:] - \
                         path_baselines[:-1]
                path["advantages"].append(special.discount_cumsum(
                    deltas, self.algo.discount * self.algo.gae_lambda))
                path["returns"].append(special.discount_cumsum(path["rewards"][i], self.algo.discount))
                baselines[i].append(path_baselines[:-1])
                returns[i].append(path["returns"][i])

        ev = [special.explained_variance_1d(
            np.concatenate(baselines[i]),
            np.concatenate(returns[i])
        ) for i in range(n)]

        if not self.algo.policy.recurrent:
            tensor_concat = lambda key: [tensor_utils.concat_tensor_list(x) for x in regroup([path[key] for path in paths])]
            tensor_concat_d = lambda key: [tensor_utils.concat_tensor_dict_list(x) for x in regroup([path[key] for path in paths])]

            observations_n = tensor_concat("observations")
            actions_n = tensor_concat("actions")
            rewards_n = tensor_concat("rewards")
            returns_n = tensor_concat("returns")
            advantages_n = tensor_concat("advantages")
            # env_infos_n = tensor_concat_d("env_infos")
            agent_infos_n = tensor_concat_d("agent_infos")

            # TODO(cathywu) make consistent with the rest (above)?
            # env_infos = tensor_utils.concat_tensor_dict_list([path["env_infos"] for path in paths])

            if self.algo.center_adv:
                advantages_n = [util.center_advantages(advantages) for advantages in advantages_n]

            if self.algo.positive_adv:
                advantages_n = [util.shift_advantages_to_positive(advantages) for advantages in advantages_n]

            average_discounted_return = \
                np.mean([sum(path["returns"][i][0] for i in range(n)) for path in paths])

            undiscounted_returns = [sum(sum(path["rewards"])) for path in paths]

            ent = np.mean(self.algo.policy.get_distribution(idx=0).entropy(agent_infos_n[0]))

            samples_data_n = [dict(
                observations=observations_n[i],
                actions=actions_n[i],
                rewards=rewards_n[i],
                returns=returns_n[i],
                advantages=advantages_n[i],
                # env_infos=env_infos,
                agent_infos=agent_infos_n[i],
                # paths=paths,
            ) for i in range(n)]
        else:
            return NotImplementedError

        logger.log("fitting baseline...")
        if hasattr(self.algo.baseline, 'fit_with_samples'):
            raise NotImplementedError
        else:
            for idx in range(len(self.algo.NPOs)):
                self.algo.NPOs[idx].baseline.fit(paths, idx=idx)
        logger.log("fitted")

        logger.record_tabular('Iteration', itr)
        logger.record_tabular('AverageDiscountedReturn',
                              average_discounted_return)
        for i in range(len(ev)):
            logger.record_tabular('ExplainedVariance-k%d' % i, ev[i])
        logger.record_tabular('NumTrajs', len(paths))
        logger.record_tabular_misc_stat('TrajLen', [len(p["rewards"][0]) for p in paths], placement='front')
        logger.record_tabular('Entropy', ent)
        logger.record_tabular('Perplexity', np.exp(ent))
        logger.record_tabular_misc_stat('Return', undiscounted_returns, placement='front')

        return samples_data_n