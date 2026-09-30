"""Remote CPU audit: warmed median timings, independent references, fixed seed."""
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
from scipy.special import gammaln  # audit only, not a package dependency
from bnn import TwoScaleBNN,bnn_weights,select_scales,two_scale_weights


def elapsed(fn, repeats=5):
    fn()
    timings=[]
    for _ in range(repeats):
        start=time.perf_counter(); fn(); timings.append(time.perf_counter()-start)
    return statistics.median(timings)


def gamma_weights(n,s):
    k=np.arange(1,n-s+2,dtype=float)
    w=np.zeros(n)
    w[:len(k)]=np.exp(gammaln(n-k+1)-gammaln(n-k-s+2)+gammaln(n-s+1)-gammaln(n+1)+np.log(s))
    return w


def baseline(X,y,Q,s1,s2):
    # Same log-gamma weighting as the elasticity and Patrick code. Recompute
    # for each scalar prediction, matching the elasticity one-query interface.
    d=X.shape[1]; a,b=s1**(-2/d),s2**(-2/d)
    out=[]
    for q in Q:
        w=-b/(a-b)*gamma_weights(len(X),s1)+a/(a-b)*gamma_weights(len(X),s2)
        idx=np.argsort(np.sum((X-q)**2,axis=1),kind="stable")
        out.append(y[idx]@w)
    return np.array(out)


def benchmark():
    rng=np.random.default_rng(20260930)
    results={"host":platform.node(),"platform":platform.platform(),"python":platform.python_version(),
             "numpy":np.__version__,"seed":20260930,"warmup":1,"repeats":5,
             "threads":{k:os.environ.get(k) for k in ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},"checks":{},"timings":[]}
    def record(name,old,new,detail):
        results["timings"].append({"name":name,"baseline_seconds":old,"optimized_seconds":new,
                                   "speedup":old/new,**detail})
    for n in (1000,10000,100000):
        s=20 if n<100000 else 100
        record("rank_weights",elapsed(lambda:gamma_weights(n,s)),elapsed(lambda:bnn_weights(n,s)),{"n":n,"s":s,"baseline":"SciPy vectorized log-gamma"})
        error=float(np.max(np.abs(bnn_weights(n,s)-gamma_weights(n,s))))
        results["checks"][f"weights_n{n}_max_abs_error"]=error
    X=rng.normal(size=(20000,4)); y=rng.normal(size=len(X)); Q=rng.normal(size=(64,4))
    model=TwoScaleBNN(20,40).fit(X,y)
    reference=baseline(X,y,Q,20,40)
    error=float(np.max(np.abs(reference-model.predict(Q))))
    assert error<1e-8,error
    results["checks"]["prediction_gamma_reference_max_abs_error"]=error
    record("64_queries",elapsed(lambda:baseline(X,y,Q,20,40)),elapsed(lambda:model.predict(Q)),{"n":len(X),"d":4,"m":64,"baseline":"scalar elasticity-style NumPy/SciPy formulation; not Numba runtime"})
    pairs=[(5,10),(10,20),(15,30),(20,40),(30,60),(40,80)]
    target=rng.normal(size=len(Q))
    def naive_tune():
        return [np.mean((TwoScaleBNN(*p).fit(X,y).predict(Q)-target)**2) for p in pairs]
    choice=select_scales(X,y,Q,target,pairs)
    np.testing.assert_allclose([s[2] for s in choice.scores],naive_tune(),rtol=1e-12,atol=1e-12)
    record("held_out_grid",elapsed(naive_tune),elapsed(lambda:select_scales(X,y,Q,target,pairs)),{"n":len(X),"d":4,"m":64,"candidates":len(pairs),"baseline":"separate optimized fit/predict per pair"})
    X=rng.normal(size=(5000,3)); y=rng.normal(size=len(X)); Q=rng.normal(size=(3,3)); B=99
    model=TwoScaleBNN(10,20).fit(X,y)
    def naive_boot():
        samples=[]; rngb=np.random.default_rng(9)
        for _ in range(B):
            idx=rngb.integers(0,len(X),size=len(X))
            samples.append(TwoScaleBNN(10,20).fit(X[idx],y[idx]).predict(Q))
        return np.array(samples)
    samples=model.bootstrap(Q,n_resamples=B,random_state=9).samples
    difference=float(np.max(np.abs(samples-naive_boot())))
    assert difference<1e-10,difference
    results["checks"]["bootstrap_max_abs_error"]=difference
    record("bootstrap",elapsed(naive_boot,3),elapsed(lambda:model.bootstrap(Q,n_resamples=B,random_state=9),3),{"n":len(X),"d":3,"m":3,"B":B,"baseline":"explicit paired resampling and optimized refits"})
    X=X[:1000]; y=y[:1000]; Q=Q[:1]; model=TwoScaleBNN(10,20).fit(X,y)
    def naive_jack():
        estimates=[]
        for i in range(len(X)):
            keep=np.arange(len(X))!=i
            estimates.append(TwoScaleBNN(10,20).fit(X[keep],y[keep]).predict(Q))
        return np.sqrt((len(X)-1)/len(X)*np.sum((np.array(estimates)-model.predict(Q))**2,axis=0))
    result=model.jackknife(Q).standard_error
    np.testing.assert_allclose(result,naive_jack(),rtol=1e-11,atol=1e-11)
    record("jackknife",elapsed(naive_jack,3),elapsed(lambda:model.jackknife(Q),3),{"n":len(X),"d":3,"m":1,"baseline":"1000 explicit delete-one refits"})
    # Rounding behavior is checked against the current Patrick C++ formula,
    # independently reproduced in Python; R itself is not installed here.
    n,d,s1,c=31,3,3,2.3; s2=int(np.ceil(c*s1))
    patrick_a=-1/(c**(2/d)-1); patrick_b=1-patrick_a
    results["checks"]["patrick_rounded_ratio_bias_residual"]=patrick_a*s1**(-2/d)+patrick_b*s2**(-2/d)
    from bnn import two_scale_coefficients
    a,b=two_scale_coefficients(d,s1,s2)
    results["checks"]["actual_integer_ratio_bias_residual"]=a*s1**(-2/d)+b*s2**(-2/d)
    Path("docs").mkdir(exist_ok=True)
    Path("docs/benchmark_cpu.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))


if __name__=="__main__": benchmark()

