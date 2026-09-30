"""Generate browser parity fixtures using the maintained Python estimators.
Run on a compute host, not the Mac cockpit.
"""
import json
from pathlib import Path
import numpy as np
from bnn import BNN, TwoScaleBNN
cases=[]
for n,s,s1,s2 in [(6,2,2,4),(12,1,1,3),(24,24,7,19)]:
    rng=np.random.default_rng(500+n)
    X=rng.uniform(-1,1,(n,1));y=rng.normal(size=n)
    if n==6: X=np.array([[0.],[0.],[1.],[1.],[2.],[3.]])
    Q=np.array([[-.8],[0.],[.5],[1.],[1.5],[2.8]])
    cases.append(dict(X=X.tolist(),y=y.tolist(),queries=Q.tolist(),s=s,s1=s1,s2=s2,
                      base=BNN(s).fit(X,y).predict(Q).tolist(),
                      two_scale=TwoScaleBNN(s1,s2).fit(X,y).predict(Q).tolist()))
Path(__file__).with_name('python-oracle.json').write_text(json.dumps(cases,indent=2)+'\n')
print('Generated',len(cases),'Python package fixtures')
