import {createServer} from 'node:http';
import {readFile, stat} from 'node:fs/promises';
import {resolve, extname, sep} from 'node:path';
const root = resolve('dist');
const types = {'.html':'text/html; charset=utf-8','.css':'text/css; charset=utf-8','.js':'text/javascript; charset=utf-8','.mjs':'text/javascript; charset=utf-8','.svg':'image/svg+xml','.png':'image/png','.pdf':'application/pdf','.json':'application/json; charset=utf-8','.bib':'text/plain; charset=utf-8','.txt':'text/plain; charset=utf-8','.xml':'application/xml; charset=utf-8','.woff2':'font/woff2','.ttf':'font/ttf'};
const server=createServer(async(req,res)=>{
  res.setHeader('X-Content-Type-Options','nosniff');
  res.setHeader('Referrer-Policy','strict-origin-when-cross-origin');
  res.setHeader('X-Frame-Options','SAMEORIGIN');
  res.setHeader('Content-Security-Policy',"default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; font-src 'self'; connect-src 'self'; base-uri 'self'; frame-ancestors 'self'; form-action 'none'");
  if(!['GET','HEAD'].includes(req.method)){res.writeHead(405,{Allow:'GET, HEAD'});res.end();return;}
  try {
    const pathname=decodeURIComponent(new URL(req.url,'http://localhost').pathname);
    if(pathname==='/healthz'){res.writeHead(200,{'Content-Type':'text/plain'});res.end(req.method==='HEAD'?undefined:'ok');return;}
    if(pathname.includes('\0')||pathname.split('/').some(p=>p.startsWith('.')&&p.length>0)){res.writeHead(404);res.end();return;}
    let file=resolve(root,'.'+pathname);
    if(file!==root&&!file.startsWith(root+sep)){res.writeHead(404);res.end();return;}
    let info;
    try {info=await stat(file);}catch {}
    if(info?.isDirectory()){
      if(!pathname.endsWith('/')){res.writeHead(308,{Location:pathname+'/'});res.end();return;}
      file=resolve(file,'index.html');
    }
    let body,status=200;
    try {body=await readFile(file);}catch {status=404;file=resolve(root,'404.html');body=await readFile(file);}
    res.writeHead(status,{'Content-Type':types[extname(file)]||'application/octet-stream','Cache-Control':extname(file)==='.html'?'no-cache':'public, max-age=300','Content-Length':body.length});
    res.end(req.method==='HEAD'?undefined:body);
  } catch {res.writeHead(400);res.end('Invalid request');}
});
server.listen(Number(process.env.PORT||8747),'0.0.0.0',()=>console.log('BNN site listening'));
