const strip = document.querySelector('.case-strip');
const group = document.querySelector('.strip-group');
// Three identical groups provide room to cross either seam without hitting
// the browser's finite scroll boundary, even with several cards in view.
for(let i=0;i<2;i++) {
  const duplicate = group.cloneNode(true);
  duplicate.setAttribute('aria-hidden','true');
  document.querySelector('.strip-track').append(duplicate);
}
const motionPreference = window.matchMedia('(prefers-reduced-motion: reduce)');
let autoplay = !motionPreference.matches;
let visible = false;
let lastTime = 0;
let frameId = 0;
let position = 0;
let manualFrame = 0;
const modulo = (value,width)=>((value%width)+width)%width;
function tick(time) {
  const delta = lastTime ? Math.min(time-lastTime,50) : 0;
  lastTime = time;
  const width = group.getBoundingClientRect().width;
  position += delta * 0.024;
  if(width) position = width + modulo(position,width);
  strip.scrollLeft = position;
  frameId = requestAnimationFrame(tick);
}
function updatePlayback() {
  cancelAnimationFrame(frameId);
  lastTime = 0;
  position = strip.scrollLeft;
  const button = document.getElementById('case-play');
  button.textContent = autoplay ? 'Pause' : 'Play';
  button.setAttribute('aria-label',autoplay ? 'Pause automatic scrolling' : 'Play automatic scrolling');
  if(autoplay && visible && !document.hidden && !strip.matches(':hover') && !strip.contains(document.activeElement)) frameId = requestAnimationFrame(tick);
}
function stopForInteraction(){cancelAnimationFrame(manualFrame);manualFrame=0;autoplay=false;updatePlayback();}
function step(direction) {
  stopForInteraction();
  const width=group.getBoundingClientRect().width;
  if(!width)return;
  let phase=modulo(strip.scrollLeft,width);
  // scrollLeft is rounded to device pixels; treat either side of the seam
  // as the same first-card position so repeated clicks cannot stall there.
  if(phase<2 || width-phase<2)phase=0;
  const origin=group.getBoundingClientRect().left;
  const offsets=[...group.querySelectorAll('figure')].map(card=>card.getBoundingClientRect().left-origin);
  // Align to actual card starts: the figures have different aspect ratios.
  const target=direction>0
    ? (offsets.find(offset=>offset>phase+2) ?? width)
    : ([...offsets].reverse().find(offset=>offset<phase-2) ?? offsets[offsets.length-1]-width);
  const from=width+phase;
  const to=width+target;
  strip.scrollLeft=from;
  const finish=()=>{strip.scrollLeft=width+modulo(to,width);position=strip.scrollLeft;manualFrame=0;};
  if(motionPreference.matches){finish();return;}
  let start;
  function animate(time) {
    if(start===undefined)start=time;
    const progress=Math.min((time-start)/420,1);
    const eased=1-Math.pow(1-progress,3);
    strip.scrollLeft=from+(to-from)*eased;
    if(progress<1)manualFrame=requestAnimationFrame(animate);
    else finish();
  }
  manualFrame=requestAnimationFrame(animate);
}
document.getElementById('case-prev').addEventListener('click',()=>step(-1));
document.getElementById('case-next').addEventListener('click',()=>step(1));
document.getElementById('case-play').addEventListener('click',()=>{autoplay=!autoplay;updatePlayback();});
['mouseenter','mouseleave','focusin'].forEach(event=>strip.addEventListener(event,updatePlayback));
strip.addEventListener('focusout',()=>queueMicrotask(updatePlayback));
strip.addEventListener('touchstart',stopForInteraction,{passive:true});
strip.addEventListener('wheel',e=>{if(Math.abs(e.deltaX)>0)stopForInteraction();},{passive:true});
strip.addEventListener('keydown',e=>{if(e.key==='ArrowLeft'||e.key==='ArrowRight'){e.preventDefault();step(e.key==='ArrowLeft'?-1:1);}});
document.addEventListener('visibilitychange',updatePlayback);
motionPreference.addEventListener('change',()=>{if(motionPreference.matches)autoplay=false;updatePlayback();});
new IntersectionObserver(entries=>{visible=entries[0].isIntersecting;updatePlayback();},{threshold:0.15}).observe(strip);
const scores={
  qwen:[[56.90,42.11,42.92,47.31],[59.99,49.01,45.13,51.38],[61.00,52.43,53.51,55.65]],
  llava:[[50.23,23.39,24.00,32.54],[50.99,30.16,34.35,38.50],[54.36,33.90,37.47,41.91]],
  intern:[[50.58,41.79,43.05,45.14],[52.08,43.10,42.69,45.96],[55.95,46.44,46.82,49.74]],
  gpt:[[21.83,28.92,40.16,30.30],[23.20,32.27,41.03,32.17],[26.79,36.18,44.29,35.75]]
};
document.getElementById('backbone').addEventListener('change',e=>{
  const rows=scores[e.target.value];
  const tbody=document.getElementById('result-rows');tbody.replaceChildren();
  rows.forEach((row,i)=>{
    const tr=document.createElement('tr');if(i===2)tr.className='ours';
    const label=document.createElement('th');label.scope='row';label.textContent=['Direct','Overview','GazeEarth'][i];tr.append(label);
    row.forEach(v=>{const td=document.createElement('td');td.textContent=v.toFixed(2);tr.append(td);});tbody.append(tr);
  });
});
document.getElementById('copy-citation').addEventListener('click',async()=>{
  const code=document.getElementById('bibtex');const status=document.getElementById('copy-status');
  try{await navigator.clipboard.writeText(code.textContent);status.textContent='Citation copied to clipboard.';}
  catch{const selection=window.getSelection();const range=document.createRange();range.selectNodeContents(code);selection.removeAllRanges();selection.addRange(range);status.textContent='Citation selected. Press Ctrl+C or ⌘C to copy.';}
});
