"""Focused adaptation of the elasticity package's bnn_2scale_torch pathway.

Keeps native PyTorch distance/order/reduction operations; caches weights and
adds bounded query batching, explicit scales, stable ties, and input checks.
"""
import torch
import numpy as np

from .core import _integer, two_scale_coefficients


def _tensor(value, name, device, dtype):
    complex_input = value.is_complex() if isinstance(value, torch.Tensor) else np.iscomplexobj(value)
    if complex_input:
        raise ValueError(f"{name} must contain real numbers")
    out = torch.as_tensor(value, device=device, dtype=dtype)
    if not bool(torch.isfinite(out).all()):
        raise ValueError(f"{name} must contain finite values")
    return out


def weight_torch(n, s, *, dtype=torch.float64, device=None):
    """Elasticity rank weights via recurrence instead of repeated log-gamma."""
    n, s = _integer(n, "n"), _integer(s, "s")
    if s > n:
        raise ValueError("s must be <= n")
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("dtype must be torch.float32 or torch.float64")
    w = torch.zeros(n, dtype=dtype, device=device)
    support = n-s+1
    w[0] = s/n
    if support > 1:
        denom = torch.arange(n-1, s-1, -1, dtype=dtype, device=device)
        w[1:support] = (s/n) * torch.cumprod(1-(s-1)/denom, dim=0)
    return w


def weight_2scale_torch(n, p, s1, s2=None, *, dtype=torch.float64, device=None):
    """Compatible elasticity-style weighting; omitted s2 means 2*s1."""
    s1 = _integer(s1, "s1")
    s2 = 2*s1 if s2 is None else _integer(s2, "s2")
    a, _ = two_scale_coefficients(p, s1, s2)
    first = weight_torch(n, s1, dtype=dtype, device=device)
    second = weight_torch(n, s2, dtype=dtype, device=device)
    return second + (-a)*(second-first)


class _TorchRankRegressor:
    """Native CPU/CUDA prediction, with explicit bounded query batch size.

    Returns a torch tensor on the selected device. This estimator is a rank
    statistic and is not offered as a differentiable neural-network layer.
    """

    def __init__(self, *, device=None, dtype=torch.float64, batch_size=8):
        if dtype not in (torch.float32, torch.float64):
            raise ValueError("dtype must be torch.float32 or torch.float64")
        self.device, self.dtype = device, dtype
        self.batch_size = _integer(batch_size, "batch_size")

    @torch.no_grad()
    def fit(self, X, y):
        device = self.device
        if device is None:
            device = X.device if isinstance(X, torch.Tensor) else "cpu"
        X = _tensor(X, "X", device, self.dtype)
        y = _tensor(y, "y", device, self.dtype)
        if X.ndim != 2 or min(X.shape) < 1:
            raise ValueError("X must be a nonempty matrix")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be a vector matching X")
        weights = self._rank_weights(len(X), X.shape[1], X.device)
        self._X, self._y, self._weights = X.detach().clone(), y.detach().clone(), weights
        self._settings = (self._parameters(), self.dtype)
        return self

    @torch.no_grad()
    def predict(self, X):
        if not hasattr(self, "_X"):
            raise ValueError("call fit before prediction")
        if self._settings != (self._parameters(), self.dtype):
            raise ValueError("scales/dtype changed after fit; call fit again")
        Q = _tensor(X, "query X", self._X.device, self.dtype)
        if Q.ndim == 1:
            Q = Q.unsqueeze(0)
        if Q.ndim != 2 or Q.shape[1] != self._X.shape[1]:
            raise ValueError("query X must have the training feature dimension")
        out = torch.empty(len(Q), dtype=self.dtype, device=self._X.device)
        for start in range(0, len(Q), self.batch_size):
            batch = Q[start:start+self.batch_size]
            # Direct differences preserve ordering at large common offsets;
            # cdist's matrix-multiply path can suffer cancellation there.
            diff = self._X.unsqueeze(0) - batch.unsqueeze(1)
            dist = torch.sum(diff.square(), dim=2)
            if not bool(torch.isfinite(dist).all()):
                raise ValueError("squared distances overflow; rescale features")
            order = torch.argsort(dist, dim=1, stable=True)
            out[start:start+len(batch)] = self._y[order] @ self._weights
        return out


class TorchBNN(_TorchRankRegressor):
    """Bagged nearest-neighbor prediction on CPU or CUDA."""

    def __init__(self, s, *, device=None, dtype=torch.float64, batch_size=8):
        self.s = _integer(s, "s")
        super().__init__(device=device, dtype=dtype, batch_size=batch_size)

    def _parameters(self):
        return (self.s,)

    def _rank_weights(self, n, d, device):
        return weight_torch(n, self.s, dtype=self.dtype, device=device)


class TorchTwoScaleBNN(_TorchRankRegressor):
    """Two-scale bias-corrected BNN prediction on CPU or CUDA."""

    def __init__(self, s1, s2, *, device=None, dtype=torch.float64, batch_size=8):
        self.s1, self.s2 = _integer(s1, "s1"), _integer(s2, "s2")
        two_scale_coefficients(1, self.s1, self.s2)
        super().__init__(device=device, dtype=dtype, batch_size=batch_size)

    def _parameters(self):
        return (self.s1, self.s2)

    def _rank_weights(self, n, d, device):
        return weight_2scale_torch(n, d, self.s1, self.s2,
                                  dtype=self.dtype, device=device)


def bnn_2scale_torch(y, X, x, s1, s2=None, *, dtype=torch.float64):
    """Drop-in one-query elasticity call; defaults to its original s2=2*s1."""
    s1 = _integer(s1, "s1")
    return TorchTwoScaleBNN(s1, 2*s1 if s2 is None else s2, dtype=dtype).fit(X, y).predict(x)[0]
