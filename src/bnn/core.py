"""Bagged nearest neighbors and its two-scale bias correction.

The base estimator averages the nearest-neighbor response over every subsample
of a specified size, using its exact rank-weight representation. The two-scale
variant follows Demirkaya et al. (2024), equations 6 and 9–11.

Only floating-point rounding is introduced: no tail truncation or approximate
neighbor search. Euclidean distances use the features supplied by the caller.
"""
from dataclasses import dataclass
from numbers import Integral
import math

import numpy as np


def _integer(value, name, minimum=1):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer >= {minimum}")
    value = int(value)
    if value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_array(value, name):
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must contain real numbers")
    try:
        out = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain real numbers") from exc
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def _training_data(X, y):
    X, y = _finite_array(X, "X"), _finite_array(y, "y")
    if X.ndim != 2 or min(X.shape) < 1:
        raise ValueError("X must be a nonempty (n_samples, n_features) matrix")
    if y.ndim != 1 or len(y) != len(X):
        raise ValueError("y must be a vector with one response per row of X")
    return X, y


def _queries(X, d):
    X = _finite_array(X, "query X")
    if X.ndim == 1:
        X = X.reshape(1, -1)
    if X.ndim != 2 or X.shape[1] != d:
        raise ValueError(f"query X must have {d} features")
    return X


def _distances(X, x):
    # Subtraction avoids the catastrophic cancellation of ||X||²-2Xx+||x||²
    # when features share a large offset. One query at a time bounds workspace.
    with np.errstate(over="ignore", invalid="ignore"):
        diff = X - x
        dist = np.einsum("ij,ij->i", diff, diff)
    if not np.isfinite(dist).all():
        raise ValueError("squared distances overflow float64; rescale features")
    return dist


def bnn_weights(n, s):
    """Rank i (1-based) receives C(n-i, s-1)/C(n,s); trailing ranks are zero.

    The recurrence avoids subtracting large log-gamma values. The returned
    length-n vector uses float64 and supports both s=1 and s=n.
    """
    n, s = _integer(n, "n"), _integer(s, "s")
    if s > n:
        raise ValueError("s must be <= n")
    w = np.zeros(n, dtype=np.float64)
    support = n - s + 1
    w[0] = s / n
    if support > 1:
        denominators = np.arange(n - 1, s - 1, -1, dtype=np.float64)
        w[1:support] = (s / n) * np.cumprod(1.0 - (s - 1) / denominators)
    return w


def two_scale_coefficients(d, s1, s2):
    """Coefficients that sum to one and cancel the s**(-2/d) bias term."""
    d, s1, s2 = _integer(d, "d"), _integer(s1, "s1"), _integer(s2, "s2")
    if s1 >= s2:
        raise ValueError("scales must satisfy s1 < s2")
    # log1p/expm1 remain accurate for nearby scales and large dimension.
    beta = 1.0 / math.expm1(2.0 / d * math.log1p((s2 - s1) / s1))
    return -beta, 1.0 + beta


def two_scale_weights(n, d, s1, s2):
    """Combined length-n rank weights of the paper's two-scale estimator."""
    n = _integer(n, "n")
    a, _ = two_scale_coefficients(d, s1, s2)
    if s2 > n:
        raise ValueError("scales must satisfy 1 <= s1 < s2 <= n")
    first, second = bnn_weights(n, s1), bnn_weights(n, s2)
    return second + (-a) * (second - first)


@dataclass(frozen=True)
class BootstrapResult:
    estimate: np.ndarray
    standard_error: np.ndarray
    samples: np.ndarray  # shape (n_resamples, n_queries); joint paired draws


@dataclass(frozen=True)
class JackknifeResult:
    estimate: np.ndarray
    standard_error: np.ndarray


@dataclass(frozen=True)
class ScaleSelection:
    s1: int
    s2: int
    mse: float
    scores: tuple  # (s1, s2, validation MSE) for each candidate in input order


class _RankRegressor:
    """Shared prediction and resampling for fixed BNN rank weights.

    Fit copies the data and caches rank weights. Predictions always have shape
    (n_queries,), including when one feature vector is supplied. Exact distance
    ties keep original training-row order (the paper's natural-index convention).
    """

    def fit(self, X, y):
        X, y = _training_data(X, y)
        weights = self._rank_weights(len(X), X.shape[1])
        self._scales = self._parameters()
        self._X, self._y = X.copy(), y.copy()
        self._weights = weights
        for data in (self._X, self._y, self._weights):
            data.setflags(write=False)
        return self

    def _check_fitted(self):
        if not hasattr(self, "_X"):
            raise ValueError("call fit before prediction or inference")
        if self._parameters() != self._scales:
            raise ValueError("scales changed after fit; call fit again")

    def predict(self, X):
        self._check_fitted()
        queries = _queries(X, self._X.shape[1])
        out = np.empty(len(queries), dtype=np.float64)
        for j, x in enumerate(queries):
            order = np.argsort(_distances(self._X, x), kind="stable")
            out[j] = self._y[order] @ self._weights
        return out

    def bootstrap(self, X, *, n_resamples=499, random_state=None):
        """Paired nonparametric bootstrap with fixed scales; variance uses B-1.

        Each query replays identical bootstrap index draws, so samples retain
        joint dependence across queries without storing a B-by-n index matrix.
        For tie-free original distances, counts and cumulative rank weights
        avoid recomputing distances and sorting every bootstrap sample. If
        distinct original observations tie, sort resampled distances stably to
        preserve their bootstrap natural-index order exactly.
        """
        self._check_fitted()
        B = _integer(n_resamples, "n_resamples", minimum=2)
        if random_state is not None:
            random_state = _integer(random_state, "random_state", minimum=0)
        queries = _queries(X, self._X.shape[1])
        n = len(self._X)
        seed = np.random.SeedSequence(random_state)
        samples = np.empty((B, len(queries)), dtype=np.float64)
        estimates = np.empty(len(queries), dtype=np.float64)
        cumulative = np.concatenate(([0.0], np.cumsum(self._weights)))
        for j, x in enumerate(queries):
            dist = _distances(self._X, x)
            order = np.argsort(dist, kind="stable")
            ordered_y = self._y[order]
            estimates[j] = ordered_y @ self._weights
            tied = np.any(np.diff(dist[order]) == 0)
            rng = np.random.default_rng(seed)
            for b in range(B):
                indices = rng.integers(0, n, size=n)
                if tied:
                    bootstrap_order = np.argsort(dist[indices], kind="stable")
                    samples[b, j] = self._y[indices[bootstrap_order]] @ self._weights
                else:
                    counts = np.bincount(indices, minlength=n)[order]
                    ends = np.cumsum(counts)
                    starts = ends - counts
                    masses = cumulative[ends] - cumulative[starts]
                    samples[b, j] = ordered_y @ masses
        return BootstrapResult(estimates, samples.std(axis=0, ddof=1), samples)

    def jackknife(self, X):
        """Exact delete-one standard errors from paper eq. 24, with fixed scales.

        Prefix/suffix sums compute every delete-one prediction after one sort.
        Every scale must be < n so it is valid in each reduced sample.
        """
        self._check_fitted()
        n = len(self._X)
        if max(self._parameters()) >= n:
            raise ValueError("jackknife requires every scale to be < n")
        queries = _queries(X, self._X.shape[1])
        reduced = self._rank_weights(n - 1, self._X.shape[1])
        estimates, errors = np.empty(len(queries)), np.empty(len(queries))
        for j, x in enumerate(queries):
            order = np.argsort(_distances(self._X, x), kind="stable")
            y = self._y[order]
            estimates[j] = y @ self._weights
            before = np.concatenate(([0.0], np.cumsum(y[:-1] * reduced)))
            after = np.concatenate((np.cumsum((y[1:] * reduced)[::-1])[::-1], [0.0]))
            leave_one = before + after
            variance = (n - 1) / n * np.sum((leave_one - estimates[j]) ** 2)
            errors[j] = np.sqrt(variance)
        return JackknifeResult(estimates, errors)


class BNN(_RankRegressor):
    """Bagged nearest-neighbor regression at subsample size ``s``.

    Uses the exact average over all size-s subsamples drawn without replacement.
    At s=1 the prediction is the sample mean; at s=n it is the nearest-neighbor
    response. Increasing s places more weight on nearby observations.
    """

    def __init__(self, s):
        self.s = _integer(s, "s")

    def _parameters(self):
        return (self.s,)

    def _rank_weights(self, n, d):
        return bnn_weights(n, self.s)


class TwoScaleBNN(_RankRegressor):
    """Bias-corrected BNN with explicit integer scales ``s1 < s2``.

    Combines two BNN estimates to cancel the leading s**(-2/d) bias term under
    the paper's assumptions. Negative weights are intentional.
    """

    def __init__(self, s1, s2):
        self.s1 = _integer(s1, "s1")
        self.s2 = _integer(s2, "s2")
        if self.s1 >= self.s2:
            raise ValueError("scales must satisfy s1 < s2")

    def _parameters(self):
        return (self.s1, self.s2)

    def _rank_weights(self, n, d):
        return two_scale_weights(n, d, self.s1, self.s2)


def select_scales(X, y, X_validation, y_validation, candidates):
    """Select explicit (s1,s2) pairs by held-out MSE, sorting each query once.

    This is ordinary held-out selection, not a reproduction of the paper's
    pointwise tuning procedure. Supply a separate validation set; no leakage
    check can infer whether the caller reused training observations.
    """
    X, y = _training_data(X, y)
    query = _queries(X_validation, X.shape[1])
    target = _finite_array(y_validation, "y_validation")
    if target.ndim != 1 or len(target) != len(query) or len(target) == 0:
        raise ValueError("y_validation must match a nonempty validation query set")
    pairs = list(candidates)
    if not pairs:
        raise ValueError("supply at least one candidate scale pair")
    weights = []
    for pair in pairs:
        if len(pair) != 2:
            raise ValueError("each candidate must be an (s1,s2) pair")
        weights.append(two_scale_weights(len(X), X.shape[1], *pair))
    weights = np.asarray(weights)
    losses = np.zeros(len(pairs))
    for x, response in zip(query, target):
        order = np.argsort(_distances(X, x), kind="stable")
        losses += (weights @ y[order] - response) ** 2
    losses /= len(query)
    best = int(np.argmin(losses))
    scores = tuple((int(a), int(b), float(loss)) for (a, b), loss in zip(pairs, losses))
    return ScaleSelection(*scores[best], scores)
