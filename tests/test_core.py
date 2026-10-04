"""Independent mathematical oracles, including exhaustive subset averaging."""
import itertools
import math
import unittest

import numpy as np
from bnn import BNN, TwoScaleBNN, bnn_weights, select_scales, two_scale_coefficients, two_scale_weights


def exact_weights(n, s):
    return np.array([math.comb(n - i, s - 1) / math.comb(n, s)
                     if n - i >= s - 1 else 0.0 for i in range(1, n + 1)])


def reference(X, y, x, s1, s2):
    order = np.argsort(np.sum((X - x) ** 2, axis=1), kind="stable")
    a, b = s1 ** (-2 / X.shape[1]), s2 ** (-2 / X.shape[1])
    return -b / (a - b) * (y[order] @ exact_weights(len(X), s1)) + a / (a - b) * (y[order] @ exact_weights(len(X), s2))


class WeightsTests(unittest.TestCase):
    def test_all_small_combinatorial_weights(self):
        for n in range(1, 40):
            for s in range(1, n + 1):
                w = bnn_weights(n, s)
                np.testing.assert_allclose(w, exact_weights(n, s), atol=2e-15, rtol=2e-13)
                self.assertAlmostEqual(w.sum(), 1.0, places=13)

    def test_bias_cancellation(self):
        for d in (1, 2, 3, 10, 100, 10000):
            for s1, s2 in ((1, 2), (3, 7), (10000, 10001)):
                a, b = two_scale_coefficients(d, s1, s2)
                self.assertLess(a, 0)
                self.assertAlmostEqual(a + b, 1)
                leading = a * s1 ** (-2/d) + b * s2 ** (-2/d)
                self.assertLess(abs(leading), 1e-14 * (abs(a) + abs(b)))

    def test_large_n_reference(self):
        # Independent log-gamma oracle (also the source package's approach).
        n, s = 100000, 31
        w = bnn_weights(n, s)
        ranks = (1, 2, 100, 1000, 10000)
        oracle = [math.exp(math.lgamma(n-i+1) - math.lgamma(n-i-s+2)
                           + math.lgamma(n-s+1) - math.lgamma(n+1) + math.log(s))
                  for i in ranks]
        np.testing.assert_allclose(w[np.array(ranks)-1], oracle, rtol=5e-9)
        self.assertAlmostEqual(w.sum(), 1, places=11)


class EstimatorTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(17)
        self.X, self.y, self.Q = rng.normal(size=(20, 3)), rng.normal(size=20), rng.normal(size=(4, 3))

    def test_exhaustive_subset_definition(self):
        X = np.array([[0.], [2.], [-2.], [4.], [6.], [9.]])
        y = np.array([2., 7., -3., 4., 20., 1.])
        x = np.array([1.])  # has distinct observations at equal distances
        values = []
        for s in (2, 4):
            picks = []
            for subset in itertools.combinations(range(len(X)), s):
                best = min(subset, key=lambda i: (float(np.sum((X[i]-x)**2)), i))
                picks.append(y[best])
            values.append(np.mean(picks))
        # d=1, s1=2,s2=4 => coefficients -1/3,4/3.
        expected = -values[0] / 3 + 4 * values[1] / 3
        result = TwoScaleBNN(2, 4).fit(X, y).predict(x)[0]
        self.assertAlmostEqual(result, expected, places=13)

    def test_dimensions_arbitrary_scales(self):
        rng = np.random.default_rng(22)
        for d in (1, 2, 3, 8):
            X, y, Q = rng.normal(size=(31, d)), rng.normal(size=31), rng.normal(size=(6, d))
            for scales in ((1, 2), (3, 7), (15, 31), (30, 31)):
                out = TwoScaleBNN(*scales).fit(X, y).predict(Q)
                expected = [reference(X, y, x, *scales) for x in Q]
                np.testing.assert_allclose(out, expected, atol=1e-12, rtol=1e-12)

    def test_constant_response_and_input_copy(self):
        X, y = self.X.copy(), np.full(20, 9.0)
        model = TwoScaleBNN(3, 7).fit(X, y)
        X[:] = 0
        y[:] = 100
        np.testing.assert_allclose(model.predict(self.Q), 9.0, atol=1e-13)

    def test_stable_ties(self):
        X, y = np.array([[-1.], [1.], [-1.], [1.]]), np.arange(4.)
        model = TwoScaleBNN(2, 4).fit(X, y)
        expected = y @ two_scale_weights(4, 1, 2, 4)
        self.assertAlmostEqual(model.predict([0.])[0], expected)

    def test_large_common_feature_offset(self):
        X, Q = self.X + 1e12, self.Q + 1e12
        expected = [reference(X, self.y, x, 3, 7) for x in Q]
        np.testing.assert_allclose(TwoScaleBNN(3, 7).fit(X, self.y).predict(Q), expected, atol=1e-12)

    def test_bootstrap_matches_explicit_resampling(self):
        for X, Q in ((self.X, self.Q), (np.array([[-1.], [1.], [2.], [-2.]]), np.array([[0.], [1.]]))):
            y = self.y[:len(X)]
            model = TwoScaleBNN(2, 4).fit(X, y)
            result = model.bootstrap(Q, n_resamples=31, random_state=71)
            rng = np.random.default_rng(71)
            expected = []
            for _ in range(31):
                idx = rng.integers(0, len(X), size=len(X))
                expected.append([reference(X[idx], y[idx], x, 2, 4) for x in Q])
            np.testing.assert_allclose(result.samples, expected, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(result.standard_error, np.std(expected, axis=0, ddof=1), atol=1e-12)
            np.testing.assert_array_equal(result.samples, model.bootstrap(Q, n_resamples=31, random_state=71).samples)

    def test_jackknife_matches_explicit_deletions(self):
        for X, y, Q in ((self.X, self.y, self.Q), (np.ones((8, 1)), np.arange(8.), np.ones((2, 1)))):
            result = TwoScaleBNN(2, 4).fit(X, y).jackknife(Q)
            leave = []
            for i in range(len(X)):
                keep = np.arange(len(X)) != i
                leave.append([reference(X[keep], y[keep], x, 2, 4) for x in Q])
            variance = (len(X)-1)/len(X) * np.sum((np.array(leave)-result.estimate)**2, axis=0)
            np.testing.assert_allclose(result.standard_error, np.sqrt(variance), atol=1e-12)

    def test_scale_selection_matches_separate_fits(self):
        pairs = [(2, 4), (3, 7), (4, 9)]
        target = np.array([1., -2., .5, 2.])
        choice = select_scales(self.X, self.y, self.Q, target, pairs)
        losses = [np.mean((TwoScaleBNN(*p).fit(self.X, self.y).predict(self.Q)-target)**2) for p in pairs]
        np.testing.assert_allclose([s[2] for s in choice.scores], losses)
        self.assertEqual((choice.s1, choice.s2), pairs[int(np.argmin(losses))])

    def test_invalid_inputs(self):
        bad = [(0, 3), (3, 3), (4, 3), (True, 3), (2.5, 4)]
        for pair in bad:
            with self.assertRaises(ValueError):
                TwoScaleBNN(*pair)
        model = TwoScaleBNN(2, 4)
        with self.assertRaises(ValueError): model.predict(self.Q)
        for X, y in ((self.X, self.y[:-1]), (np.array([[np.nan]]), np.array([1.])), (np.ones((3,1)), np.ones(3)), (self.X.astype(complex), self.y)):
            with self.assertRaises(ValueError): model.fit(X,y)
        model.fit(self.X, self.y)
        for Q in (np.ones((2, 2)), np.array([np.inf, 1, 2])):
            with self.assertRaises(ValueError): model.predict(Q)
        for B in (0, 1, 2.5, True):
            with self.assertRaises(ValueError): model.bootstrap(self.Q, n_resamples=B)
        with self.assertRaises(ValueError): model.bootstrap(self.Q, random_state=-1)
        model.s1 = 3
        with self.assertRaises(ValueError): model.predict(self.Q)
        with self.assertRaises(ValueError): TwoScaleBNN(2, 20).fit(self.X, self.y).jackknife(self.Q)
        with self.assertRaises(ValueError): TwoScaleBNN(1, 2).fit([[1e300], [-1e300]], [1.,2.]).predict([0.])
        with self.assertRaises(ValueError): select_scales(self.X, self.y, self.Q, np.ones(4), [])


class BaseBNNTests(unittest.TestCase):
    def setUp(self):
        self.X = np.array([[-2.], [2.], [-1.], [1.], [0.], [3.]])
        self.y = np.array([7., -2., 5., 11., -4., 3.])
        self.Q = np.array([[0.], [1.], [-1.], [.25]])

    @staticmethod
    def subset_average(X, y, Q, s):
        return np.array([
            np.mean([
                y[min(subset, key=lambda i: (float(np.sum((X[i]-x)**2)), i))]
                for subset in itertools.combinations(range(len(X)), s)
            ]) for x in Q
        ])

    def test_base_definition_and_endpoints(self):
        for s in range(1, len(self.X)+1):
            expected = self.subset_average(self.X, self.y, self.Q, s)
            np.testing.assert_allclose(BNN(s).fit(self.X, self.y).predict(self.Q), expected, atol=1e-13)
        np.testing.assert_allclose(BNN(1).fit(self.X, self.y).predict(self.Q), self.y.mean())
        np.testing.assert_array_equal(BNN(1).fit([[0.]], [3.]).predict([[0.], [20.]]), [3., 3.])

    def test_base_bootstrap_against_subset_averages(self):
        # Includes distance ties and both endpoint scales.
        for s in (1, 3, len(self.X)):
            result = BNN(s).fit(self.X, self.y).bootstrap(self.Q, n_resamples=19, random_state=41)
            rng = np.random.default_rng(41)
            expected = []
            for _ in range(19):
                indices = rng.integers(0, len(self.X), size=len(self.X))
                expected.append(self.subset_average(self.X[indices], self.y[indices], self.Q, s))
            np.testing.assert_allclose(result.samples, expected, atol=1e-13)
            np.testing.assert_allclose(result.standard_error, np.std(expected, axis=0, ddof=1), atol=1e-13)

    def test_base_jackknife_against_subset_averages(self):
        for s in (1, 3, len(self.X)-1):
            model = BNN(s).fit(self.X, self.y)
            result = model.jackknife(self.Q)
            leave = []
            for i in range(len(self.X)):
                keep = np.arange(len(self.X)) != i
                leave.append(self.subset_average(self.X[keep], self.y[keep], self.Q, s))
            variance = (len(self.X)-1)/len(self.X)*np.sum((leave-result.estimate)**2, axis=0)
            np.testing.assert_allclose(result.standard_error, np.sqrt(variance), atol=1e-13)

    def test_base_validation_and_fitted_cache(self):
        for s in (0, -1, True, 2.5):
            with self.assertRaises(ValueError): BNN(s)
        with self.assertRaises(ValueError): BNN(7).fit(self.X, self.y)
        with self.assertRaises(ValueError): BNN(6).fit(self.X, self.y).jackknife(self.Q)
        model = BNN(2)
        with self.assertRaises(ValueError): model.predict(self.Q)
        model.fit(self.X, self.y)
        model.s = 3
        with self.assertRaises(ValueError): model.predict(self.Q)
        np.testing.assert_allclose(model.fit(self.X, self.y).predict(self.Q),
                                   self.subset_average(self.X, self.y, self.Q, 3))


if __name__ == "__main__":
    unittest.main()
