"""One-price, one-instrument elasticity extension of Wang and Huang.

The prediction method is supplied by the caller. The price equation is fitted
once by least squares, and five query predictions share the fitted BNN weights.
This is the focused scalar pathway; paper-specific multidimensional replication
remains in replication/elasticity.
"""
from dataclasses import dataclass
import numpy as np
from .core import _integer, _queries, _training_data


def _column(index, d, name):
    if isinstance(index, (bool, np.bool_)) or not isinstance(index, (int, np.integer)):
        raise ValueError(f"{name} must be an integer column index")
    if index < -d or index >= d:
        raise ValueError(f"{name} is outside the feature dimension")
    return int(index) % d


def _host(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


@dataclass(frozen=True)
class ElasticityResult:
    elasticity: np.ndarray
    uncorrected_elasticity: np.ndarray
    prediction: np.ndarray
    price_derivative: np.ndarray
    instrument_derivative: np.ndarray
    first_stage_derivative: np.ndarray


class PriceElasticity:
    """Scalar price elasticity using a BNN regressor and instrument correction.

    ``step`` is the total span of a centered finite difference. The first stage
    regresses price on an intercept, instrument powers, and optional controls.
    Its defaults reproduce the quadratic scalar pathway in the elasticity code.
    Estimates have the structural interpretation only under the paper's
    identification conditions. Neither regression accuracy nor a nonsingular
    first stage establishes instrument validity.
    """

    def __init__(self, regressor, *, price_index=0, instrument_index=-1,
                 step=0.1, first_stage_degree=2, first_stage_controls=(),
                 min_first_stage=1e-8):
        self.regressor = regressor
        self.price_index, self.instrument_index = price_index, instrument_index
        self.step = float(step)
        self.min_first_stage = float(min_first_stage)
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValueError("step must be positive and finite")
        if not np.isfinite(self.min_first_stage) or self.min_first_stage <= 0:
            raise ValueError("min_first_stage must be positive and finite")
        self.first_stage_degree = _integer(first_stage_degree, "first_stage_degree")
        self.first_stage_controls = tuple(first_stage_controls)

    def fit(self, X, quantity):
        X, quantity = _training_data(X, quantity)
        d = X.shape[1]
        p = _column(self.price_index, d, "price_index")
        z = _column(self.instrument_index, d, "instrument_index")
        controls = tuple(_column(i, d, "control index") for i in self.first_stage_controls)
        if p == z or len(set(controls)) != len(controls) or p in controls or z in controls:
            raise ValueError("price, instrument, and control columns must be distinct")
        with np.errstate(over="ignore", invalid="ignore"):
            powers = [X[:, z]**k for k in range(1, self.first_stage_degree+1)]
            design = np.column_stack([np.ones(len(X)), *powers, *[X[:, i] for i in controls]])
        if not np.isfinite(design).all():
            raise ValueError("first-stage features overflow; rescale the instrument")
        coefficients, _, rank, _ = np.linalg.lstsq(design, X[:, p], rcond=None)
        if rank < design.shape[1]:
            raise ValueError("first-stage design is rank deficient")
        self.regressor.fit(X, quantity)
        self._coefficients, self._d, self._p, self._z = coefficients, d, p, z
        self._settings = (self.price_index, self.instrument_index, self.first_stage_degree,
                          self.first_stage_controls)
        return self

    def predict(self, X):
        if not hasattr(self, "_coefficients"):
            raise ValueError("call fit before prediction")
        settings = (self.price_index, self.instrument_index, self.first_stage_degree,
                    self.first_stage_controls)
        if settings != self._settings:
            raise ValueError("first-stage settings changed after fit; call fit again")
        if not np.isfinite(self.step) or self.step <= 0:
            raise ValueError("step must be positive and finite")
        if not np.isfinite(self.min_first_stage) or self.min_first_stage <= 0:
            raise ValueError("min_first_stage must be positive and finite")
        Q = _queries(X, self._d)
        xp, xm, zp, zm = (Q.copy() for _ in range(4))
        xp[:, self._p] += self.step/2
        xm[:, self._p] -= self.step/2
        zp[:, self._z] += self.step/2
        zm[:, self._z] -= self.step/2
        values = _host(self.regressor.predict(np.concatenate([Q, xp, xm, zp, zm])))
        base, pxp, pxm, pzp, pzm = np.split(values, 5)
        dp, dz = (pxp-pxm)/self.step, (pzp-pzm)/self.step
        with np.errstate(over="ignore", invalid="ignore"):
            first = sum(k*self._coefficients[k]*Q[:, self._z]**(k-1)
                        for k in range(1, self.first_stage_degree+1))
        if not np.isfinite(first).all() or np.any(np.abs(first) < self.min_first_stage):
            raise ValueError("first-stage derivative is nonfinite or below min_first_stage")
        if np.any(base <= 0):
            raise ValueError("predicted quantity must be positive to compute elasticity")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            uncorrected = dp*Q[:, self._p]/base
            corrected = (dp+dz/first)*Q[:, self._p]/base
        if not np.isfinite(corrected).all() or not np.isfinite(uncorrected).all():
            raise ValueError("elasticity calculation overflowed; rescale features")
        return ElasticityResult(corrected, uncorrected, base, dp, dz, first)
