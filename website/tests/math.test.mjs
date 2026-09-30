import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {rankWeights,coefficients,predict,twoScale,combinedWeights,generator,subsample,samplePrediction} from '../public/assets/math.mjs';
function close(a,b,tolerance=1e-12){assert.ok(Math.abs(a-b)<tolerance,'Expected '+a+' near '+b);}
function choose(n,k){if(k<0||k>n)return 0n;let value=1n;for(let i=1;i<=k;i++)value=value*BigInt(n-i+1)/BigInt(i);return value;}
test('rank recurrence matches independent binomial ratios and expected rank',()=>{
 for(let n=1;n<=35;n++)for(let s=1;s<=n;s++){
  const w=rankWeights(n,s);
  w.forEach((v,i)=>close(v,Number(choose(n-i-1,s-1))/Number(choose(n,s))));
  close(w.reduce((a,b)=>a+b,0),1);
  close(w.reduce((a,b,i)=>a+b*(i+1),0),(n+1)/(s+1));
 }
});
function combinations(n,s,start=0,items=[],result=[]){if(!s){result.push([...items]);return result;}for(let i=start;i<=n-s;i++)combinations(n,s-1,i+1,[...items,i],result);return result;}
test('weighted prediction equals exhaustive averaging, including distance ties',()=>{
 const points=[{x:0,y:-2},{x:1,y:4},{x:1,y:-3},{x:2,y:7},{x:3,y:1},{x:4,y:8}];
 for(const x of [0,.5,1,1.5,3.5])for(let s=1;s<=points.length;s++){
  const values=combinations(points.length,s).map(sub=>{
   const sorted=sub.sort((a,b)=>(points[a].x-x)**2-(points[b].x-x)**2||a-b);return points[sorted[0]].y;
  });
  close(predict(points,x,s),values.reduce((a,b)=>a+b,0)/values.length);
 }
});
test('endpoints, constants and signed coefficient identities',()=>{
 const p=[{x:0,y:1},{x:1,y:5},{x:2,y:9}];
 close(predict(p,.8,1),5);close(predict(p,.8,3),5);
 for(const d of [1,2,4,7])for(const [s1,s2] of [[1,2],[4,9],[20,21]]){
  const [a,b]=coefficients(s1,s2,d);close(a+b,1);close(a*s1**(-2/d)+b*s2**(-2/d),0);
  close(combinedWeights(24,s1,s2,d).reduce((x,y)=>x+y,0),1);
 }
 close(twoScale(p.map(x=>({...x,y:3})),.7,1,3,1),3);
});
test('draws contain distinct observations and use the correct stable nearest response',()=>{
 const p=[{x:0,y:1},{x:0,y:9},{x:1,y:4},{x:2,y:-2}];const random=generator(4);
 for(let i=0;i<200;i++){
  const draw=samplePrediction(p,0,3,random);
  assert.equal(new Set(draw.chosen).size,3);
  const best=[...draw.chosen].sort((a,b)=>p[a].x**2-p[b].x**2||a-b)[0];
  assert.equal(draw.best,best);assert.equal(draw.value,p[best].y);
 }
 assert.deepEqual(subsample(5,3,generator(7)),subsample(5,3,generator(7)));
});
test('browser estimates agree with independently generated Python package fixtures',()=>{
 const cases=JSON.parse(readFileSync(new URL('./python-oracle.json',import.meta.url),'utf8'));
 for(const c of cases){
  const p=c.X.map((x,i)=>({x:x[0],y:c.y[i]}));
  for(let i=0;i<c.queries.length;i++){
   close(predict(p,c.queries[i][0],c.s),c.base[i]);
   close(twoScale(p,c.queries[i][0],c.s1,c.s2,1),c.two_scale[i]);
  }
 }
});
test('invalid scale inputs fail clearly',()=>{
 for(const args of [[0,1],[4,0],[4,5],[4,1.2]])assert.throws(()=>rankWeights(...args),RangeError);
 assert.throws(()=>coefficients(2,2,1),RangeError);
});
