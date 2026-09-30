import {toyData,generator,order,rankWeights,predict,samplePrediction,regressionFunction,coefficients,combinedWeights,twoScale} from './math.mjs';

const points=toyData();
const ns='http://www.w3.org/2000/svg';
function node(svg,type,attrs={},text){
  const e=document.createElementNS(ns,type);
  for(const [name,value] of Object.entries(attrs))e.setAttribute(name,String(value));
  if(text!==undefined)e.textContent=text;svg.append(e);return e;
}
function frame(svg,{min=-1.15,max=1.35}={}){
  svg.replaceChildren();
  const W=680,H=320,left=55,right=650,top=28,bottom=267;
  const X=x=>left+x*(right-left),Y=y=>bottom-(y-min)/(max-min)*(bottom-top);
  for(let i=0;i<=4;i++){
    const y=min+(max-min)*i/4;
    node(svg,'path',{d:'M'+left+' '+Y(y)+'H'+right,class:'gridline'});
    node(svg,'text',{x:left-10,y:Y(y)+4,'text-anchor':'end'},y.toFixed(1));
  }
  node(svg,'path',{d:'M'+left+' '+top+'V'+bottom+'H'+right,class:'axis'});
  for(const x of [0,.25,.5,.75,1])node(svg,'text',{x:X(x),y:bottom+22,'text-anchor':'middle'},x.toFixed(2));
  node(svg,'text',{x:352,y:312,'text-anchor':'middle'},'Feature x');
  node(svg,'text',{x:17,y:145,transform:'rotate(-90 17 145)','text-anchor':'middle'},'Response');
  return {X,Y,left,right,top,bottom,min,max};
}
function path(svg,values,f,className){
  node(svg,'path',{d:values.map((p,i)=>(i?'L':'M')+f.X(p.x).toFixed(2)+' '+f.Y(p.y).toFixed(2)).join(' '),class:className});
}
const queries=Array.from({length:181},(_,i)=>i/180);
function drawWeights(svg,weights,signed=false){
  svg.replaceChildren();
  const left=40,right=654,top=18,bottom=signed?125:105;
  const max=Math.max(.0001,...weights),min=signed?Math.min(0,...weights):0;
  const Y=y=>bottom-(y-min)/(max-min)*(bottom-top);
  const zero=Y(0),width=(right-left)/weights.length;
  node(svg,'path',{d:'M'+left+' '+zero+'H'+right,class:'axis'});
  weights.forEach((weight,i)=>{
    node(svg,'rect',{x:left+i*width+2,y:Math.min(Y(weight),zero),width:Math.max(1,width-4),height:Math.max(.5,Math.abs(Y(weight)-zero)),class:weight<0?'negative-weight':'weight'});
    if([0,5,11,17,23].includes(i))node(svg,'text',{x:left+(i+.5)*width,y:bottom+19,'text-anchor':'middle',class:'small-label'},String(i+1));
  });
  node(svg,'text',{x:left-7,y:Y(max)+4,'text-anchor':'end',class:'small-label'},max.toFixed(2));
  if(signed&&min<0)node(svg,'text',{x:left-7,y:Y(min)+4,'text-anchor':'end',class:'small-label'},min.toFixed(2));
  node(svg,'text',{x:right,y:bottom+34,'text-anchor':'end',class:'small-label'},'Distance rank');
}
const demo=document.querySelector('#bagging-demo');
if(demo){
  const svg=demo.querySelector('#bagging-plot');
  const query=demo.querySelector('#query');
  const scale=demo.querySelector('#scale-input');
  const play=demo.querySelector('#play');
  const status=demo.querySelector('#demo-status');
  const motion=matchMedia('(prefers-reduced-motion: reduce)');
  const state={x:.48,s:5,count:0,total:0,draw:null,random:generator(712),timer:null,curve:null};
  function announce(text){status.textContent=text;}
  function stop(){clearInterval(state.timer);state.timer=null;play.textContent=motion.matches?'Draw 80 subsamples':'Play 80 draws';}
  function render(){
    const f=frame(svg);
    if(!state.curve)state.curve=queries.map(x=>({x,y:predict(points,x,state.s)}));
    path(svg,queries.map(x=>({x,y:regressionFunction(x)})),f,'truth-line');
    path(svg,state.curve,f,'exact-line');
    node(svg,'path',{d:'M'+f.X(state.x)+' '+f.top+'V'+f.bottom,class:'query-line'});
    points.forEach((p,i)=>node(svg,'circle',{cx:f.X(p.x),cy:f.Y(p.y),r:state.draw?.best===i?7:5.5,class:state.draw?.best===i?'nearest':state.draw?.chosen.includes(i)?'selected':'observation'}));
    const exact=predict(points,state.x,state.s);
    node(svg,'circle',{cx:f.X(state.x),cy:f.Y(exact),r:7,fill:'var(--exact)',stroke:'white','stroke-width':2});
    if(state.count){
      const avg=state.total/state.count;
      node(svg,'path',{d:'M'+(f.X(state.x)-28)+' '+f.Y(avg)+'h56',class:'sample-line'});
    }
    demo.querySelector('#query-label').textContent=state.x.toFixed(2);
    demo.querySelector('#scale-label').textContent=state.s+' of '+points.length;
    demo.querySelector('#exact-value').textContent=exact.toFixed(3);
    demo.querySelector('#sample-value').textContent=state.count?(state.total/state.count).toFixed(3):'No draws yet';
    demo.querySelector('#draw-count').textContent=String(state.count);
    demo.querySelector('#rank-mean').textContent='Expected rank '+((points.length+1)/(state.s+1)).toFixed(2);
    svg.setAttribute('aria-label','Synthetic regression, query '+state.x.toFixed(2)+', scale '+state.s+'. Exact BNN '+exact.toFixed(3)+(state.count?', sampled mean '+(state.total/state.count).toFixed(3)+' after '+state.count+' draws':'. No subsamples drawn.'));
    drawWeights(demo.querySelector('#weights-plot'),rankWeights(points.length,state.s));
  }
  function clear(){
    stop();state.count=0;state.total=0;state.draw=null;state.random=generator(712);
  }
  function step(){
    state.draw=samplePrediction(points,state.x,state.s,state.random);state.total+=state.draw.value;state.count++;render();
  }
  query.addEventListener('input',()=>{clear();state.x=Number(query.value);render();});
  scale.addEventListener('input',()=>{clear();state.s=Number(scale.value);state.curve=null;render();});
  demo.querySelector('#draw-one').addEventListener('click',()=>{stop();step();announce('Draw '+state.count+'. Sample mean '+(state.total/state.count).toFixed(3)+'.');});
  demo.querySelector('#reset').addEventListener('click',()=>{clear();render();announce('Subsample average reset.');});
  play.addEventListener('click',()=>{
    if(state.timer){stop();announce('Paused after '+state.count+' draws.');return;}
    if(motion.matches){
      for(let i=0;i<80;i++){state.draw=samplePrediction(points,state.x,state.s,state.random);state.total+=state.draw.value;state.count++;}
      render();announce('Drew 80 subsamples. Total '+state.count+'.');return;
    }
    const end=state.count+80;play.textContent='Pause';announce('Playing 80 subsample draws.');
    state.timer=setInterval(()=>{step();if(state.count>=end){stop();announce('Finished '+state.count+' subsample draws.');}},180);
  });
  document.addEventListener('visibilitychange',()=>{if(document.hidden)stop();});
  new IntersectionObserver(entries=>{if(!entries[0].isIntersecting)stop();},{threshold:.1}).observe(demo);
  motion.addEventListener('change',stop);
  render();stop();demo.classList.add('ready');
}
const combined=document.querySelector('#two-scale-demo');
if(combined){
  const first=combined.querySelector('#s1'),second=combined.querySelector('#s2');
  function render(changed){
    let s1=Number(first.value),s2=Number(second.value);
    if(s1>=s2){
      if(changed==='first'){s2=Math.min(points.length,s1+1);second.value=String(s2);}
      else{s1=Math.max(1,s2-1);first.value=String(s1);}
    }
    combined.querySelector('#s1-label').textContent=String(s1);combined.querySelector('#s2-label').textContent=String(s2);
    const base=queries.map(x=>({x,y:predict(points,x,s1)})),two=queries.map(x=>({x,y:twoScale(points,x,s1,s2,1)}));
    const values=[...points,...base,...two];
    const min=Math.min(-1.15,...values.map(p=>p.y))-.06,max=Math.max(1.35,...values.map(p=>p.y))+.06;
    const svg=combined.querySelector('#two-plot'),f=frame(svg,{min,max});
    path(svg,queries.map(x=>({x,y:regressionFunction(x)})),f,'truth-line');path(svg,base,f,'sample-line');path(svg,two,f,'exact-line');
    points.forEach(p=>node(svg,'circle',{cx:f.X(p.x),cy:f.Y(p.y),r:4.7,class:'observation'}));
    const [a,b]=coefficients(s1,s2,1);
    combined.querySelector('#coefficients').textContent='Coefficients: a₁ = '+a.toFixed(3)+', a₂ = '+b.toFixed(3)+'. Positive and negative bars show the combined rank weights.';
    drawWeights(combined.querySelector('#signed-weights'),combinedWeights(points.length,s1,s2,1),true);
    svg.setAttribute('aria-label','Same synthetic sample, base scale '+s1+' and two-scale combination with scales '+s1+' and '+s2+'.');
  }
  first.addEventListener('input',()=>render('first'));second.addEventListener('input',()=>render('second'));render();combined.classList.add('ready');
}
