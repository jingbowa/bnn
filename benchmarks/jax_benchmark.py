"""JAX compile and warmed prediction times, on the selected compute host."""
import json
from pathlib import Path
import platform
import statistics
import time
import jax
import numpy as np
from bnn import TwoScaleBNN

jax.config.update("jax_enable_x64",True)
from bnn.jax_backend import JAXTwoScaleBNN

rng=np.random.default_rng(71)
X=rng.normal(size=(20000,4)); y=rng.normal(size=20000); Q=rng.normal(size=(64,4))
model=JAXTwoScaleBNN(20,40,batch_size=8).fit(X,y)
begin=time.perf_counter(); value=model.predict(Q); value.block_until_ready()
first=time.perf_counter()-begin
expected=TwoScaleBNN(20,40).fit(X,y).predict(Q)
error=float(np.max(np.abs(np.asarray(value)-expected))); assert error<1e-10,error
durations=[]
for _ in range(5):
    begin=time.perf_counter(); model.predict(Q).block_until_ready(); durations.append(time.perf_counter()-begin)
result={"host":platform.node(),"jax":jax.__version__,"devices":[str(d) for d in jax.devices()],
        "n":20000,"m":64,"d":4,"dtype":"float64","first_call_seconds":first,
        "warm_median_seconds":statistics.median(durations),"repeats":5,"numpy_max_abs_error":error,
        "note":"CPU-tested JAX; GPU and TPU JAX runtimes not tested"}
Path("docs/benchmark_jax.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result,indent=2))
