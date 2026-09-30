const menu=document.querySelector('.menu-button');
const nav=document.querySelector('#navigation');
menu?.addEventListener('click',()=>{
  const open=menu.getAttribute('aria-expanded')!=='true';
  menu.setAttribute('aria-expanded',String(open));nav.classList.toggle('is-open',open);
});
document.addEventListener('keydown',e=>{if(e.key==='Escape'&&menu?.getAttribute('aria-expanded')==='true'){menu.setAttribute('aria-expanded','false');nav.classList.remove('is-open');menu.focus();}});
for(const pre of document.querySelectorAll('pre')){
  const button=document.createElement('button');button.className='copy-button';button.textContent='Copy code';button.type='button';
  button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText(pre.querySelector('code').textContent);button.textContent='Copied';setTimeout(()=>button.textContent='Copy code',1600);}catch{button.textContent='Select to copy';const selection=getSelection();const range=document.createRange();range.selectNodeContents(pre.querySelector('code'));selection.removeAllRanges();selection.addRange(range);}});
  pre.append(button);
}
