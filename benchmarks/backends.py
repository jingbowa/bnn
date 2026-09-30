"""Compare new backend with the actual elasticity PyTorch TDNN routine."""
import importlib.util
import json
import os
from pathlib import Path
import platform
import statistics
import time

import numpy as np
import torch
from bnn import TwoScaleBNN
from bnn.torch_backend import TorchTwoScaleBNN


def run():
    device="cuda" if os.environ.get("BNN_TEST_CUDA") else "cpu"
    if device=="cuda": assert torch.cuda.is_available()
    torch.set_num_threads(4)
    reference_file=os.environ.get("BNN_ELASTICITY_TORCH")
    legacy=None
    if reference_file:
        spec=importlib.util.spec_from_file_location("elasticity_reference",reference_file)
        legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
    rng=np.random.default_rng(71)
    X=rng.normal(size=(20000,4)); y=rng.normal(size=20000); Q=rng.normal(size=(64,4))
    tx,ty,tq=[torch.tensor(v,dtype=torch.float64,device=device) for v in (X,y,Q)]
    model=TorchTwoScaleBNN(20,40,device=device,batch_size=8).fit(tx,ty)
    expected=TwoScaleBNN(20,40).fit(X,y).predict(Q)
    output=model.predict(tq).cpu().numpy()
    error=float(np.max(np.abs(output-expected))); assert error<1e-10,error
    results={"host":platform.node(),"torch":torch.__version__,"device":device,"n":20000,"m":64,"d":4,
             "dtype":"float64","threads":4,"warmup":1,"repeats":5,"numpy_max_abs_error":error}
    if device=="cuda":
        results["gpu"]=torch.cuda.get_device_name(0)
        torch.cuda.reset_peak_memory_stats()
    def timed(fn):
        fn()
        if device=="cuda": torch.cuda.synchronize()
        durations=[]
        for _ in range(5):
            if device=="cuda": torch.cuda.synchronize()
            begin=time.perf_counter(); fn()
            if device=="cuda": torch.cuda.synchronize()
            durations.append(time.perf_counter()-begin)
        return statistics.median(durations)
    results["optimized_seconds"]=timed(lambda:model.predict(tq))
    if legacy:
        def original():
            return torch.stack([legacy.bnn_2scale_torch(ty,tx,q,20) for q in tq])
        baseline=original().cpu().numpy()
        diff=float(np.max(np.abs(baseline-output))); assert diff<1e-8,diff
        results["elasticity_max_abs_error"]=diff
        results["baseline_seconds"]=timed(original)
        results["speedup"]=results["baseline_seconds"]/results["optimized_seconds"]
        results["baseline"]="actual elasticity bnn_2scale_torch, one call per query"
    if device=="cuda": results["peak_allocated_mib"]=torch.cuda.max_memory_allocated()/2**20
    if device=="cpu":
        try:
            import jax
            jax.config.update("jax_enable_x64",True)
            from bnn.jax_backend import JAXTwoScaleBNN
            jm=JAXTwoScaleBNN(20,40,batch_size=8).fit(X,y)
            first=time.perf_counter(); value=jm.predict(Q); value.block_until_ready()
            results["jax_first_call_seconds"]=time.perf_counter()-first
            diff=float(np.max(np.abs(np.asarray(value)-expected))); assert diff<1e-10,diff
            results["jax_numpy_max_abs_error"]=diff
            results["jax_version"]=jax.__version__
            results["jax_devices"]=[str(d) for d in jax.devices()]
            results["jax_warm_seconds"]=timed(lambda:jm.predict(Q).block_until_ready())
        except ImportError:
            results["jax"]="not installed"
    Path("docs").mkdir(exist_ok=True)
    Path(f"docs/benchmark_{device}_backends.json").write_text(json.dumps(results,indent=2)+"\n")
    print(json.dumps(results,indent=2))

if __name__=="__main__": run()
