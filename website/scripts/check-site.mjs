import {readFile,readdir,stat} from 'node:fs/promises';
import {resolve,join} from 'node:path';
import assert from 'node:assert/strict';
const root=resolve('dist');
async function walk(dir){let result=[];for(const name of await readdir(dir)){const path=join(dir,name);if((await stat(path)).isDirectory())result.push(...await walk(path));else if(path.endsWith('.html'))result.push(path);}return result;}
const files=await walk(root);
for(const file of files){
 const html=await readFile(file,'utf8');
 assert.equal((html.match(/<h1[ >]/g)||[]).length,1,file);
 for(const m of html.matchAll(/(?:href|src)="([^"]+)"/g)){
  if(!m[1].startsWith('/')||m[1].startsWith('//'))continue;
  const [path,anchor]=m[1].split('#');let target=join(root,path);
  const info=await stat(target);if(info.isDirectory())target=join(target,'index.html');
  await stat(target);
  if(anchor&&target.endsWith('.html')){
   const dest=await readFile(target,'utf8');
   assert.ok(dest.includes('id="'+anchor+'"'),m[1]+' in '+file);
  }
 }
}
console.log('Checked all internal links/assets/anchors in '+files.length+' pages');
