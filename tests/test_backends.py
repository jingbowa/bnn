import importlib.util
import os
import unittest
import numpy as np
from bnn import BNN, TwoScaleBNN

HAS_TORCH = importlib.util.find_spec("torch") is not None
HAS_JAX = importlib.util.find_spec("jax") is not None


def cases():
    rng = np.random.default_rng(314)
    for d in (1, 2, 5):
        yield rng.normal(size=(41, d)), rng.normal(size=41), rng.normal(size=(11, d))
    yield np.array([[-1.],[1.],[-1.],[1.]]), np.arange(4.), np.array([[0.],[1.],[-1.]])
    X = rng.normal(size=(32, 2)) + 1e12
    yield X, rng.normal(size=32), X[:9] + .1


@unittest.skipUnless(HAS_TORCH, "PyTorch not installed")
class TorchTests(unittest.TestCase):
    def compare(self, device, dtype, tolerance):
        import torch
        from bnn.torch_backend import TorchBNN, TorchTwoScaleBNN, bnn_2scale_torch, weight_2scale_torch
        for X,y,Q in cases():
            if dtype == torch.float32 and np.max(np.abs(X)) > 1e10:
                continue  # float32 cannot represent these input distinctions
            scales = (2,4) if len(X)==4 else (3,7)
            expected = TwoScaleBNN(*scales).fit(X,y).predict(Q)
            for batch_size in (1,4,16):
                model = TorchTwoScaleBNN(*scales, device=device, dtype=dtype, batch_size=batch_size).fit(X,y)
                np.testing.assert_allclose(model.predict(Q).cpu().numpy(),expected,rtol=tolerance,atol=tolerance)
                self.assertEqual(len(model.predict(Q[:0])),0)
                for s in (1, 3, len(X)):
                    base = TorchBNN(s, device=device, dtype=dtype, batch_size=batch_size).fit(X,y)
                    np.testing.assert_allclose(base.predict(Q).cpu().numpy(), BNN(s).fit(X,y).predict(Q),
                                               rtol=tolerance, atol=tolerance)
            # Compatibility wrapper checks the original ratio-two interface.
            tx,ty,tq = (torch.tensor(v,device=device,dtype=dtype) for v in (X,y,Q[0]))
            result = bnn_2scale_torch(ty,tx,tq,2,dtype=dtype)
            target = TwoScaleBNN(2,4).fit(X,y).predict(Q[0])[0]
            self.assertAlmostEqual(result.item(),target,delta=tolerance*max(1,abs(target)))
            w=weight_2scale_torch(len(X),X.shape[1],*scales,dtype=dtype,device=device)
            self.assertAlmostEqual(w.sum().item(),1.,delta=tolerance)

    def test_cpu_float64(self):
        import torch
        self.compare("cpu",torch.float64,2e-12)

    def test_cpu_float32(self):
        import torch
        self.compare("cpu",torch.float32,3e-5)

    def test_cuda_float64(self):
        import torch
        if not os.environ.get("BNN_TEST_CUDA"):
            self.skipTest("CUDA tests require explicit BNN_TEST_CUDA after VRAM gate")
        self.assertTrue(torch.cuda.is_available())
        self.compare("cuda",torch.float64,2e-11)

    def test_cuda_float32(self):
        import torch
        if not os.environ.get("BNN_TEST_CUDA"):
            self.skipTest("CUDA tests require explicit BNN_TEST_CUDA after VRAM gate")
        self.assertTrue(torch.cuda.is_available())
        self.compare("cuda",torch.float32,3e-5)

    def test_validation(self):
        import torch
        from bnn.torch_backend import TorchTwoScaleBNN
        with self.assertRaises(ValueError): TorchTwoScaleBNN(2,4,dtype=torch.float16)
        with self.assertRaises(ValueError): TorchTwoScaleBNN(2,4).fit(np.ones((3,1)),np.ones(3))
        model=TorchTwoScaleBNN(2,4).fit(np.ones((5,2)),np.ones(5))
        with self.assertRaises(ValueError): model.predict([[np.inf,1.]])
        with self.assertRaises(ValueError): model.predict([[1.]])


@unittest.skipUnless(HAS_JAX, "JAX not installed")
class JAXTests(unittest.TestCase):
    def test_float64_parity(self):
        import jax
        jax.config.update("jax_enable_x64", True)
        from bnn.jax_backend import JAXBNN,JAXTwoScaleBNN,bnn_2_jax
        for X,y,Q in cases():
            scales=(2,4) if len(X)==4 else (3,7)
            expected=TwoScaleBNN(*scales).fit(X,y).predict(Q)
            for batch_size in (1,4,16):
                model=JAXTwoScaleBNN(*scales,batch_size=batch_size).fit(X,y)
                np.testing.assert_allclose(np.asarray(model.predict(Q)),expected,rtol=2e-11,atol=2e-11)
                self.assertEqual(len(model.predict(Q[:0])),0)
                for s in (1,3,len(X)):
                    base=JAXBNN(s,batch_size=batch_size).fit(X,y)
                    np.testing.assert_allclose(np.asarray(base.predict(Q)),BNN(s).fit(X,y).predict(Q),rtol=2e-11,atol=2e-11)
            target=TwoScaleBNN(2,4).fit(X,y).predict(Q[0])[0]
            self.assertAlmostEqual(float(bnn_2_jax(y,X,Q[0],2)),target,places=11)

    def test_float32_parity(self):
        from bnn.jax_backend import JAXBNN,JAXTwoScaleBNN
        for X,y,Q in cases():
            if np.max(np.abs(X))>1e10: continue
            expected=TwoScaleBNN(2,4).fit(X,y).predict(Q)
            np.testing.assert_allclose(np.asarray(JAXTwoScaleBNN(2,4,dtype="float32").fit(X,y).predict(Q)),expected,rtol=3e-5,atol=3e-5)
            for s in (1,3,len(X)):
                np.testing.assert_allclose(np.asarray(JAXBNN(s,dtype="float32").fit(X,y).predict(Q)),
                                           BNN(s).fit(X,y).predict(Q),rtol=3e-5,atol=3e-5)

    def test_requires_explicit_x64(self):
        import jax
        from bnn.jax_backend import JAXTwoScaleBNN
        previous=jax.config.x64_enabled
        try:
            jax.config.update("jax_enable_x64",False)
            with self.assertRaises(ValueError): JAXTwoScaleBNN(2,4)
            JAXTwoScaleBNN(2,4,dtype="float32")
        finally:
            jax.config.update("jax_enable_x64",previous)

if __name__=="__main__": unittest.main()
