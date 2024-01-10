from torch.utils.data import Dataset
from torch import nn
from scipy.spatial.transform import Rotation as R

import torch
import numpy as np

# all defaults are made with assumed 20hz sampling

V_MAX = 150


class BaseDataset(Dataset):
    def __init__(self, bag, start=0, end=None):
        if not end:
            end = len(bag["t"])

        self.name = bag["name"]

        self.p = bag["p"][start:end]
        self.v = bag["v"][start:end]  # replace with wheel speed
        self.u = bag["u"][start:end]
        self.r = bag["r"][start:end]

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        return (
            self.p[i],
            self.v[i],
            self.u[i],
            self.r[i],
        )


class ContDataset(BaseDataset):
    def __init__(
        self, bag, window=1, horizon=100, delta_p=True, theta=True, fill_holes=True
    ):  # single step window, 5 sec horizon
        super().__init__(bag)

        self.p = np.array(self.p)
        self.v = np.array(self.v)
        self.u = np.array(self.u)
        self.r = np.array(self.r)

        self.window = window
        self.horizon = horizon
        self.delta_p = delta_p
        self.theta = theta

        # standardize_u_and_v
        self.u_mean, self.u_std = None, None
        self.v_mean, self.v_std = None, None
        self.standardize_warning = True
        # normalize_u
        self.p_min, self.p_r = None, None
        self.normalize_warning = True

        if fill_holes:
            self.p = self._fill_holes(self.p)

    def __len__(self):
        return self.r.shape[0] - self.window - self.horizon

    def __getitem__(self, i):
        j = i + self.window + self.horizon
        v = np.copy(self.v[i:j])
        u, r = (
            np.copy(self.u[i : j - 1]),
            np.copy(self.r[i : j - 1]),
        )
        v_x, v_y = np.copy(self.v[i : j - 1]), np.copy(self.v[i + 1 : j])
        p_x, p_y = np.copy(self.p[i : j - 1]), np.copy(self.p[i + 1 : j])

        if self.u_mean is None and self.standardize_warning:
            print(
                "Standardization factors for u are not set. Continuing with unstandardized values."
            )
            self.standardize_warning = False
        elif self.u_mean is not None:
            u = (u - self.u_mean) / self.u_std

        if self.p_min is None and self.normalize_warning:
            print(
                "Normalization values for p are not set. Continuing with unnormalized values."
            )
            self.normalize_warning = False
        elif self.p_min is not None:
            p_x = (p_x - self.p_min) / self.p_r
            p_y = (p_y - self.p_min) / self.p_r

        # normalize v
        v_x = v_x / V_MAX
        v_y = v_y / V_MAX

        if self.delta_p:
            p_y[1:] -= p_y[:-1]  # delta p
            p_y[0] -= p_x[0]

        if self.theta:
            r = np.expand_dims(R.from_matrix(r).as_euler("zyx")[:, 0], axis=-1)

        return np.hstack([p_x, v_x, u, r]), np.hstack([p_y, v_y])

    def set_u_standardization_factors(self, u_mean, u_std):
        self.u_mean, self.u_std = u_mean, u_std

    def get_u_standardization_factors(self):
        u_mean, u_std = np.mean(self.u, axis=0), np.std(self.u, axis=0)
        return u_mean, u_std, self.p.shape[0]

    def set_p_normalization_factors(self, p_min, p_r):
        self.p_min, self.p_r = p_min, p_r

    def get_p_normalization_factors(self):
        return np.min(self.p, axis=0), np.max(self.p, axis=0) - np.min(self.p, axis=0)

    def _fill_holes(self, p):
        for i in range(1, p.shape[0] - 1):
            if np.all(np.isclose(p[i - 1], p[i])):
                p[i] = (p[i - 1] + p[i + 1]) / 2

        if np.isclose(p[-1], p[-2]).all():
            p[-1] = p[-2] - p[-3] + p[-1]

        return p


class DiscDataset(ContDataset):
    def __init__(
        self, bag, os_dim_size=1000, p_unit_size=0.25, v_max=24, u_max=12, horizon=100
    ):  # window = 1 only supported
        super().__init__(bag, window=1, horizon=horizon, delta_p=False)

        self.os_dim_size = os_dim_size
        self.p_unit_size = p_unit_size
        self.v_max = v_max
        self.u_max = u_max

        # disable normalize warning
        self.normalize_warning = False

        # values
        self.p = self.p[:, :2]
        self.v = self.v[:, :2]
        self.u = self.u[:, :2]
        self.r = np.expand_dims(R.from_matrix(self.r).as_euler("zyx")[:, 0], axis=-1)

        # norm
        self.min, self.max = None, None

    def __getitem__(self, i):
        j = i + self.window + self.horizon
        p, v, u, r = (
            np.copy(self.p[i:j]),
            np.copy(self.v[i:j]),
            np.copy(self.u[i:j]),
            np.copy(self.r[i:j]),
        )

        p0 = np.copy(p[0])
        p -= p0

        discretize = (
            lambda x, step: np.clip(
                np.abs(x) // step * np.where(x < 0, -1, 1) + np.where(x < 0, -1, 0),
                -self.os_dim_size,
                self.os_dim_size - 1,
            )
            + self.os_dim_size
        )  # uniform buckets

        p = discretize(p, self.p_unit_size)
        v = discretize(v, (self.v_max / self.os_dim_size))
        u = discretize(u, (self.u_max / self.os_dim_size))
        r = discretize(r, (np.pi / self.os_dim_size))

        p[0] = (self.p[i] - self.p_min / self.p_r) // (self.os_dim_size * 2)  # context

        # return p, self.p[i:j] - p0
        return np.hstack([p, v, u, r])  # single output for teacher forcing


def set_global_p_normalization_factors(datasets):
    v = np.vstack(
        [
            np.expand_dims(np.array(dataset.get_p_normalization_factors()), 0)
            for dataset in datasets
        ]
    )
    mi = np.min(v[:, 0], axis=0)
    ma = np.max(v[:, 1], axis=0)

    for dataset in datasets:
        dataset.set_p_normalization_factors(mi, ma - mi)

    return mi, ma - mi


def set_global_u_standardization_factors(datasets):
    u_m, u_v, t = 0, 0, 0
    for d in datasets:
        _u_m, _u_s, _t = d.get_u_standardization_factors()
        u_m += _u_m * _t
        u_v += _u_s**2 * _t
        t += _t
    u_m /= t
    u_s = np.sqrt(u_v / t)
    for d in datasets:
        d.set_u_standardization_factors(u_m, u_s)
    return u_m, u_s
