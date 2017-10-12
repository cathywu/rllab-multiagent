from rllab.baselines.base import Baseline
from rllab.misc.overrides import overrides
import numpy as np


class LinearFeatureBaseline(Baseline):
    def __init__(self, env_spec, reg_coeff=1e-5):
        self._coeffs = None
        self._reg_coeff = reg_coeff

    @overrides
    def get_param_values(self, **tags):
        return self._coeffs

    @overrides
    def set_param_values(self, val, **tags):
        self._coeffs = val

    def _features(self, path, idx=None):
        if idx is not None:
            o = np.clip(path["observations"][idx], -10, 10)
            l = len(path["rewards"][idx])
        else:
            o = np.clip(path["observations"], -10, 10)
            l = len(path["rewards"])

        al = np.arange(l).reshape(-1, 1) / 100.0
        return np.concatenate([o, o ** 2, al, al ** 2, al ** 3, np.ones((l, 1))], axis=1)

    @overrides
    def fit(self, paths, idx=None):
        featmat = np.concatenate([self._features(path, idx=idx) for path in paths])
        if idx is not None:
            returns = np.concatenate([path["returns"][idx] for path in paths])
        else:
            returns = np.concatenate([path["returns"] for path in paths])
        reg_coeff = self._reg_coeff
        for _ in range(5):
            self._coeffs = np.linalg.lstsq(
                featmat.T.dot(featmat) + reg_coeff * np.identity(featmat.shape[1]),
                featmat.T.dot(returns)
            )[0]
            if not np.any(np.isnan(self._coeffs)):
                break
            reg_coeff *= 10

    @overrides
    def predict(self, path, idx=None):
        import ipdb; ipdb.set_trace()
        if self._coeffs is None:
            if idx is not None:
                return np.zeros(len(path["rewards"][idx]))
            else:
                return np.zeros(len(path["rewards"]))
        return self._features(path, idx=idx).dot(self._coeffs)
