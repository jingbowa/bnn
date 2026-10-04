"""Focused adaptation of the legacy elasticity bnn_vec_2_jax pathway.

The elasticity vmap/jit approach is retained, with cached valid rank weights,
bounded query batches, explicit integer scales, and no global config mutation.
"""
import jax
import jax.numpy as jnp
import numpy as np

from .core import _integer, _queries, _training_data, bnn_weights, two_scale_weights


@jax.jit
def _prediction_kernel(X, y, Q, weights):
    def one(x):
        diff = X-x
        distance = jnp.sum(jnp.square(diff), axis=1)
        order = jnp.argsort(distance, stable=True)
        prediction = jnp.dot(y[order], weights)
        return prediction, jnp.all(jnp.isfinite(distance))
    return jax.vmap(one)(Q)


class _JAXRankRegressor:
    """JIT/vmap prediction on a caller-selected JAX device (CPU, GPU, or TPU).

    Input validation and weight construction happen once on the host. Prediction
    batches run on the selected device. Set jax_enable_x64 explicitly before
    requesting float64; this module never changes the application's global config.
    """

    def __init__(self, *, device=None, dtype="float64", batch_size=8):
        self.dtype = np.dtype(dtype)
        if self.dtype not in (np.dtype("float32"), np.dtype("float64")):
            raise ValueError("dtype must be float32 or float64")
        if self.dtype == np.dtype("float64") and not jax.config.x64_enabled:
            raise ValueError("float64 requires jax.config.update('jax_enable_x64', True)")
        self.device = device
        self.batch_size = _integer(batch_size, "batch_size")

    def fit(self, X, y):
        X, y = _training_data(X, y)
        weights = self._rank_weights(len(X), X.shape[1])
        with np.errstate(over="ignore", invalid="ignore"):
            X, y = np.array(X, dtype=self.dtype), np.array(y, dtype=self.dtype)
        if not np.isfinite(X).all() or not np.isfinite(y).all():
            raise ValueError("training values overflow the selected dtype")
        self._X = jax.device_put(X, self.device)
        self._y = jax.device_put(y, self.device)
        self._weights = jax.device_put(weights.astype(self.dtype), self.device)
        self._settings = (self._parameters(), self.dtype)
        return self

    def predict(self, X):
        if not hasattr(self, "_X"):
            raise ValueError("call fit before prediction")
        if self._settings != (self._parameters(), self.dtype):
            raise ValueError("scales/dtype changed after fit; call fit again")
        queries = _queries(X, self._X.shape[1])
        if not len(queries):
            return jax.device_put(np.empty(0, dtype=self.dtype), self.device)
        results = []
        for start in range(0, len(queries), self.batch_size):
            Q = jax.device_put(queries[start:start+self.batch_size].astype(self.dtype), self.device)
            values, finite = _prediction_kernel(self._X, self._y, Q, self._weights)
            if not bool(np.asarray(finite).all()):
                raise ValueError("squared distances overflow; rescale features")
            results.append(values)
        return jnp.concatenate(results)


class JAXBNN(_JAXRankRegressor):
    """Bagged nearest-neighbor prediction on the selected JAX device."""

    def __init__(self, s, *, device=None, dtype="float64", batch_size=8):
        self.s = _integer(s, "s")
        super().__init__(device=device, dtype=dtype, batch_size=batch_size)

    def _parameters(self):
        return (self.s,)

    def _rank_weights(self, n, d):
        return bnn_weights(n, self.s)


class JAXTwoScaleBNN(_JAXRankRegressor):
    """Two-scale bias-corrected BNN on the selected JAX device."""

    def __init__(self, s1, s2, *, device=None, dtype="float64", batch_size=8):
        self.s1, self.s2 = _integer(s1, "s1"), _integer(s2, "s2")
        if self.s1 >= self.s2:
            raise ValueError("scales must satisfy s1 < s2")
        super().__init__(device=device, dtype=dtype, batch_size=batch_size)

    def _parameters(self):
        return (self.s1, self.s2)

    def _rank_weights(self, n, d):
        return two_scale_weights(n, d, self.s1, self.s2)


def bnn_2_jax(y, X, x, s1, s2=None, *, dtype="float64", device=None):
    """Compatible legacy elasticity one-query call, with default s2=2*s1."""
    s1 = _integer(s1, "s1")
    return JAXTwoScaleBNN(s1, 2*s1 if s2 is None else s2, dtype=dtype, device=device).fit(X, y).predict(x)[0]
