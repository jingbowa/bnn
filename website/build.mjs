import {readFile,writeFile,mkdir,cp,rm} from 'node:fs/promises';
import {join} from 'node:path';
const settings=JSON.parse(await readFile('src/site.json','utf8'));
const origin=(process.env.SITE_URL||settings.origin).replace(/\/$/,'');
const routes=[
 ['', 'Bagged nearest neighbors', 'Understand bagged nearest neighbors through exact rank weights, interactive explanations, statistical theory, algorithms, and reproducible software.'],
 ['explore','Explore the estimator','An interactive explanation of BNN: subsampling, nearest neighbors, exact rank weights, scale choice, and two-scale bias correction.'],
 ['theory','Statistical theory','Exact representation, bias expansions, pointwise and uniform inference, derivative and quotient estimation, and bootstrap results for BNN.'],
 ['algorithms','Algorithms and computation','Exact BNN algorithms, rank-weight recurrence, computational reuse, regression inference, and documented computation times.'],
 ['examples','Statistical examples','A reproducible regression illustration and a guide to interpreting estimation error, uncertainty, and uniformity.'],
 ['software','Software','Install the BNN Python package, run regression and inference, and choose NumPy, PyTorch, or JAX prediction backends.'],
 ['papers','Papers and references','Scientific references, full author attribution, earlier working papers, and citation guidance for bagged nearest neighbors.']
];
await rm('dist',{recursive:true,force:true});await mkdir('dist',{recursive:true});await cp('public','dist',{recursive:true});
const layout=await readFile('src/layout.html','utf8');
const escape=s=>s.replaceAll('&','&amp;').replaceAll('"','&quot;').replaceAll('<','&lt;').replaceAll('>','&gt;');
function nav(active){return routes.slice(1).map(([slug])=>'<a href="/'+slug+'/"'+(slug===active?' aria-current="page"':'')+'>'+({explore:'Explore',theory:'Theory',algorithms:'Algorithms',examples:'Examples',software:'Software',papers:'Papers'}[slug])+'</a>').join('');}
const benchmarks=JSON.parse(await readFile('public/assets/benchmarks.json','utf8'));
const current=benchmarks.current.map(r=>'<tr><th scope="row">'+r.label+'</th><td>'+r.seconds.toFixed(4)+' s</td><td>'+r.context+'</td></tr>').join('');
const historical=benchmarks.historical.map(r=>'<tr><th scope="row">'+r.n.toLocaleString('en-US')+'</th><td>'+r.numba+'</td><td>'+r.cpu+'</td><td>'+r.gpu+'</td></tr>').join('');
for(const [slug,title,description] of routes){
  const raw=await readFile('src/pages/'+(slug||'home')+'.html','utf8');
  const content=raw.replaceAll('{{ current_timings }}',current).replaceAll('{{ paper_timings }}',historical);
  const values={origin,title:escape(title+' | BNN'),description:escape(description),canonical:origin+(slug?'/'+slug+'/':'/'),nav:nav(slug),content,scripts:slug==='explore'?'<script type="module" src="/assets/explore.js"></script>':''};
  const html=layout.replace(/{{ (\w+) }}/g,(_,key)=>values[key]??'');
  const directory=join('dist',slug);await mkdir(directory,{recursive:true});await writeFile(join(directory,'index.html'),html);
}
const notfound=layout.replace(/{{ (\w+) }}/g,(_,key)=>({origin,title:'Page not found | BNN',description:'This BNN page could not be found.',canonical:origin+'/404.html',nav:nav(''),content:'<header class="page-heading"><p class="page-label">404</p><h1>That page could not be found.</h1><p>Return to <a href="/">bagged nearest neighbors</a> or <a href="/explore/">explore the estimator</a>.</p></header>',scripts:''}[key]??''));
await writeFile('dist/404.html',notfound);
await writeFile('dist/sitemap.xml','<?xml version="1.0" encoding="UTF-8"?><urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'+routes.map(([slug])=>'<url><loc>'+origin+(slug?'/'+slug+'/':'/')+'</loc></url>').join('')+'</urlset>');
await writeFile('dist/robots.txt','User-agent: *\nAllow: /\nSitemap: '+origin+'/sitemap.xml\n');
console.log('Built '+routes.length+' BNN pages');
