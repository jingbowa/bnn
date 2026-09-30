import unittest
import numpy as np
from bnn import TwoScaleBNN
from bnn.elasticity import PriceElasticity


class LinearDemand:
    """Known conditional mean: structural slope -2 and control effect +3*u."""
    def fit(self, X, y):
        return self

    def predict(self, X):
        return 20-2*X[:, 0]+3*(X[:, 0]-(2+1.5*X[:, -1]+.25*X[:, -1]**2))


class ElasticityTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(187)
        z = rng.uniform(-1, 1, 300)
        u = rng.normal(0, .2, 300)
        p = 2+1.5*z+.25*z*z+u
        self.X = np.column_stack([p, z])
        self.y = LinearDemand().predict(self.X)

    def test_known_structural_correction(self):
        # Make the first-stage population coefficients exact in the sample.
        design = np.column_stack([np.ones(len(self.X)), self.X[:, 1], self.X[:, 1]**2])
        residual = self.X[:, 0]-design@np.linalg.lstsq(design, self.X[:, 0], rcond=None)[0]
        X = self.X.copy()
        X[:, 0] = 2+1.5*X[:, 1]+.25*X[:, 1]**2+residual
        model = PriceElasticity(LinearDemand()).fit(X, self.y)
        Q = np.array([[2., 0.], [2.8, .5]])
        result = model.predict(Q)
        expected_quantity = LinearDemand().predict(Q)
        np.testing.assert_allclose(result.elasticity, -2*Q[:, 0]/expected_quantity, atol=1e-13)
        np.testing.assert_allclose(result.uncorrected_elasticity, Q[:, 0]/expected_quantity, atol=1e-13)

    def test_matches_explicit_scalar_pipeline(self):
        model = PriceElasticity(TwoScaleBNN(7,14)).fit(self.X, self.y)
        Q = np.array([[2., 0.], [2.8, .5]])
        result = model.predict(Q)
        design = np.column_stack([np.ones(len(self.X)), self.X[:, 1], self.X[:, 1]**2])
        coefficients = np.linalg.lstsq(design, self.X[:, 0], rcond=None)[0]
        expected = []
        for x in Q:
            reg = TwoScaleBNN(7,14).fit(self.X, self.y)
            xp,xm,zp,zm = (x.copy() for _ in range(4))
            xp[0]+=.05; xm[0]-=.05; zp[1]+=.05; zm[1]-=.05
            dp=(reg.predict(xp)[0]-reg.predict(xm)[0])/.1
            dz=(reg.predict(zp)[0]-reg.predict(zm)[0])/.1
            first=coefficients[1]+2*coefficients[2]*x[1]
            expected.append((dp+dz/first)*x[0]/reg.predict(x)[0])
        np.testing.assert_allclose(result.elasticity,expected,atol=1e-13)

    def test_validation(self):
        for step in (0,-1,np.inf,np.nan):
            with self.assertRaises(ValueError): PriceElasticity(LinearDemand(),step=step)
        with self.assertRaises(ValueError): PriceElasticity(LinearDemand(),price_index=1).fit(self.X,self.y)
        with self.assertRaises(ValueError): PriceElasticity(LinearDemand()).fit(np.ones((10,2)),np.ones(10))
        with self.assertRaises(ValueError): PriceElasticity(LinearDemand(),min_first_stage=100).fit(self.X,self.y).predict([2,0])
        model=PriceElasticity(LinearDemand()).fit(self.X,self.y)
        model.first_stage_degree=1
        with self.assertRaises(ValueError): model.predict([2,0])


if __name__ == '__main__': unittest.main()
