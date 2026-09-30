import importlib.util
import os
import unittest
import numpy as np
from bnn import PriceElasticity, TwoScaleBNN


class ElasticityBackendTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(193)
        z = rng.uniform(-1, 1, 200)
        p = 2+1.5*z+.25*z*z+rng.normal(0, .2, len(z))
        X = np.column_stack([p, rng.uniform(-1, 1, len(z)), z])
        y = 20-2*p+3*(p-(2+1.5*z+.25*z*z))
        Q = np.array([[2., .2, 0.], [2.8, -.2, .5]])
        return X, y, Q

    def check(self, regressor):
        X, y, Q = self.data()
        actual = PriceElasticity(regressor).fit(X, y).predict(Q)
        expected = PriceElasticity(TwoScaleBNN(7,14)).fit(X,y).predict(Q)
        for field in actual.__dataclass_fields__:
            np.testing.assert_allclose(getattr(actual, field), getattr(expected, field),
                                       rtol=1e-10, atol=1e-10)

    @unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch not installed')
    def test_torch_cpu_wrapper(self):
        from bnn.torch_backend import TorchTwoScaleBNN
        self.check(TorchTwoScaleBNN(7,14,device='cpu'))

    @unittest.skipUnless(importlib.util.find_spec('torch') and os.environ.get('BNN_TEST_CUDA'),
                         'CUDA test requires explicit VRAM gate')
    def test_torch_cuda_wrapper(self):
        from bnn.torch_backend import TorchTwoScaleBNN
        self.check(TorchTwoScaleBNN(7,14,device='cuda'))

    @unittest.skipUnless(importlib.util.find_spec('jax'), 'JAX not installed')
    def test_jax_cpu_wrapper(self):
        import jax
        jax.config.update('jax_enable_x64', True)
        from bnn.jax_backend import JAXTwoScaleBNN
        self.check(JAXTwoScaleBNN(7,14,device=jax.devices('cpu')[0]))


if __name__ == '__main__': unittest.main()
