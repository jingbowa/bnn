export function rankWeights(n,s){
  if(!Number.isInteger(n)||!Number.isInteger(s)||n<1||s<1||s>n)throw new RangeError('Use integer scales with 1 <= s <= n.');
  const weights=Array(n).fill(0);weights[0]=s/n;
  for(let i=1;i<n-s+1;i++)weights[i]=weights[i-1]*(n-i-s+1)/(n-i);
  return weights;
}
export function order(points,query){return points.map((p,i)=>({i,d:(p.x-query)**2})).sort((a,b)=>a.d-b.d||a.i-b.i).map(p=>p.i);}
export function predict(points,query,s){const idx=order(points,query),w=rankWeights(points.length,s);return idx.reduce((total,i,r)=>total+w[r]*points[i].y,0);}
export function coefficients(s1,s2,d){
  if(!Number.isInteger(s1)||!Number.isInteger(s2)||s1<1||s2<=s1||!Number.isFinite(d)||d<1)throw new RangeError('Use ordered scales and positive dimension.');
  const a1=-1/Math.expm1(2*Math.log1p((s2-s1)/s1)/d);return [a1,1-a1];
}
export function combinedWeights(n,s1,s2,d){const [a,b]=coefficients(s1,s2,d),u=rankWeights(n,s1),v=rankWeights(n,s2);return u.map((x,i)=>a*x+b*v[i]);}
export function twoScale(points,query,s1,s2,d=1){const idx=order(points,query),w=combinedWeights(points.length,s1,s2,d);return idx.reduce((total,i,r)=>total+w[r]*points[i].y,0);}
export function generator(seed=2026){let a=seed>>>0;return ()=>{a+=0x6D2B79F5;let t=a;t=Math.imul(t^t>>>15,t|1);t^=t+Math.imul(t^t>>>7,t|61);return ((t^t>>>14)>>>0)/4294967296;};}
export function subsample(n,s,random){if(s<1||s>n||!Number.isInteger(s))throw new RangeError('Invalid subsample');const idx=Array.from({length:n},(_,i)=>i);for(let i=0;i<s;i++){const j=i+Math.floor(random()*(n-i));[idx[i],idx[j]]=[idx[j],idx[i]];}return idx.slice(0,s);}
export function samplePrediction(points,query,s,random){const chosen=subsample(points.length,s,random);const best=chosen.reduce((a,b)=>(points[b].x-query)**2<(points[a].x-query)**2||((points[b].x-query)**2===(points[a].x-query)**2&&b<a)?b:a);return {chosen,best,value:points[best].y};}
export function regressionFunction(x){return Math.sin(2*Math.PI*x)*.75+.4*x;}
export function toyData(seed=2026,n=24){const random=generator(seed);return Array.from({length:n},(_,i)=>{const x=.035+i*.93/(n-1);return {x,y:regressionFunction(x)+(random()-.5)*.65};});}
