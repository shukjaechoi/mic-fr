const colors=['#d5f36a','#ff9c72','#72d7df','#66e0a4','#31b6a4','#a88cff','#f3c96a','#e789b5','#8db8ff'];
const $=id=>document.getElementById(id);
let data,active=0,reference=3,selected=new Set(),buffers=[],regionStart=0,regionEnd=0;
let hoveredMic=null;
let waveformEnvelope=[],waveformDrag=null;
let context,source,sourceGain,gain,playing=false,position=0,startedAt=0;
function fmt(t){return `${Math.floor(t/60)}:${String(Math.floor(t%60)).padStart(2,'0')}`}
function time(){if(!playing)return position;const t=position+context.currentTime-startedAt;return source?.loop&&t>=regionEnd?regionStart+(t-regionEnd)%(regionEnd-regionStart):Math.min(t,regionEnd)}
function regionPct(t){return `${Math.max(0,Math.min(100,t/data.durationSec*100))}%`}
function syncTimeline(){if(!data)return;const now=time();$('current').textContent=fmt(now);$('seek').value=Math.round(now/data.durationSec*1000);$('waveform-playhead').style.left=regionPct(now);$('waveform-region').style.left=regionPct(regionStart);$('waveform-region').style.width=regionPct(regionEnd-regionStart);document.querySelectorAll('.track-playhead').forEach(el=>el.style.left=regionPct(now));document.querySelectorAll('.track-region').forEach(el=>{el.style.left=regionPct(regionStart);el.style.width=regionPct(regionEnd-regionStart)});}
function buildWaveform(buffer){const samples=buffer.getChannelData(0),bins=1200;waveformEnvelope=Array.from({length:bins},(_,i)=>{const start=Math.floor(i*samples.length/bins),end=Math.floor((i+1)*samples.length/bins),step=Math.max(1,Math.floor((end-start)/256));let sum=0,count=0;for(let j=start;j<end;j+=step){sum+=samples[j]*samples[j];count++}return Math.sqrt(sum/Math.max(1,count))});drawWaveform()}
function drawWaveform(){if(!waveformEnvelope.length)return;const canvas=$('waveform-canvas'),rect=canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;const dpr=window.devicePixelRatio||1,w=Math.round(rect.width*dpr),h=Math.round(rect.height*dpr);canvas.width=w;canvas.height=h;const g=canvas.getContext('2d');g.clearRect(0,0,w,h);const sorted=[...waveformEnvelope].sort((a,b)=>a-b),scale=sorted[Math.floor(sorted.length*.98)]||1,mid=h/2;g.fillStyle='#87ad78';for(let x=0;x<w;x++){const a=Math.floor(x/w*waveformEnvelope.length),b=Math.max(a+1,Math.floor((x+1)/w*waveformEnvelope.length));let level=0;for(let i=a;i<b;i++)level=Math.max(level,waveformEnvelope[i]||0);const bar=Math.max(1,Math.min(h*.46,level/scale*h*.43));g.fillRect(x,mid-bar,1,bar*2)}}
function waveformTime(clientX){const rect=$('waveform').getBoundingClientRect();return Math.max(0,Math.min(data.durationSec,(clientX-rect.left)/rect.width*data.durationSec))}
function setWaveformRegion(start,end){$('region-start').value=Math.round(start/data.durationSec*1000);$('region-end').value=Math.round(end/data.durationSec*1000);updateRegion()}
function trackRow(i){const m=data.microphones[i],row=document.createElement('div');row.className='track-row'+(i===active?' active':'');row.innerHTML=`<span class="track-number">${String(i+1).padStart(2,'0')}</span><span class="swatch" style="background:${colors[i]}"></span><span class="track-name">${m.label}</span><button class="solo-button" type="button" aria-pressed="${i===active}" aria-label="${m.label} Solo">SOLO</button><button class="track-timeline" type="button" aria-label="${m.label} 타임라인에서 재생 위치 선택"><span class="track-region"></span><span class="track-playhead"></span></button>`;row.querySelector('.solo-button').addEventListener('click',()=>switchMic(i));row.querySelector('.track-timeline').addEventListener('click',e=>{const r=e.currentTarget.getBoundingClientRect();seekTo((e.clientX-r.left)/r.width*data.durationSec)});return row}
function renderTracks(){$('mic-buttons').replaceChildren(...data.microphones.map((_,i)=>trackRow(i)));syncTimeline()}
function stopSource(fadeSeconds=0){if(!source)return;const old=source,oldGain=sourceGain,now=context.currentTime;old.onended=null;source=null;sourceGain=null;if(fadeSeconds>0){oldGain.gain.cancelScheduledValues(now);oldGain.gain.setValueAtTime(oldGain.gain.value,now);oldGain.gain.linearRampToValueAtTime(0,now+fadeSeconds);old.stop(now+fadeSeconds);old.addEventListener('ended',()=>{old.disconnect();oldGain.disconnect()},{once:true})}else{old.stop();old.disconnect();oldGain.disconnect()}}
function startSource(fadeSeconds=0){if(!buffers[active])return;stopSource();const from=Math.max(regionStart,Math.min(position,regionEnd-.01)),now=context.currentTime;position=from;startedAt=now;source=context.createBufferSource();sourceGain=context.createGain();source.buffer=buffers[active];source.connect(sourceGain);sourceGain.connect(gain);sourceGain.gain.setValueAtTime(fadeSeconds>0?0:1,now);if(fadeSeconds>0)sourceGain.gain.linearRampToValueAtTime(1,now+fadeSeconds);if($('loop').checked){source.loop=true;source.loopStart=regionStart;source.loopEnd=regionEnd;source.start(0,from)}else{source.onended=()=>{source=null;sourceGain?.disconnect();sourceGain=null;playing=false;position=regionStart;syncPlayButton();syncTimeline()};source.start(0,from,regionEnd-from)}playing=true;syncPlayButton()}
async function play(){if(!context||!buffers[active])return;await context.resume();if(position>=regionEnd-.01||position<regionStart)position=regionStart;startSource()}
function pause(){if(!playing)return;position=time();playing=false;stopSource();syncPlayButton();syncTimeline()}
function switchMic(i){if(i===active)return;const wasPlaying=playing;position=time();playing=false;stopSource(wasPlaying ? .018 : 0);active=i;if(wasPlaying)startSource(.018);renderTracks();syncPlayButton()}
function syncPlayButton(){$('play').textContent=playing?'❚❚':'▶';$('play').setAttribute('aria-label',playing?'일시정지':'재생')}
function seekTo(seconds){const wasPlaying=playing;playing=false;stopSource();position=Math.max(0,Math.min(data.durationSec-.01,seconds));if(wasPlaying)startSource();syncTimeline()}
function updateRegion(changed){const wasPlaying=playing,now=time(),duration=data.durationSec,minGap=.25;regionStart=+$('region-start').value/1000*duration;regionEnd=+$('region-end').value/1000*duration;if(regionEnd-regionStart<minGap){if(changed==='start')regionStart=Math.max(0,regionEnd-minGap);else regionEnd=Math.min(duration,regionStart+minGap)}$('region-start').value=Math.round(regionStart/duration*1000);$('region-end').value=Math.round(regionEnd/duration*1000);$('start-value').textContent=fmt(regionStart);$('end-value').textContent=fmt(regionEnd);$('region-summary').textContent=`${fmt(regionStart)} – ${fmt(regionEnd)}`;if(wasPlaying){playing=false;stopSource();position=now>=regionStart&&now<regionEnd?now:regionStart;startSource()}syncTimeline()}
function playRegion(){playing=false;stopSource();position=regionStart;syncTimeline();play()}
function tick(){if(data&&playing)syncTimeline();requestAnimationFrame(tick)}
function pill(v){return `<span class="pill ${v>0?'pos':v<0?'neg':''}">${v>0?'+':''}${v.toFixed(1)}</span>`}
function drawTable(){$('readings').innerHTML=data.microphones.map((m,i)=>`<tr><td><span class="swatch" style="background:${colors[i]};display:inline-block;margin-right:9px"></span>${m.label}${m.polarityInverted?' ↕':''}</td><td>${pill(m.bands.bass)}</td><td>${pill(m.bands.lowMid)}</td><td>${pill(m.bands.presence)}</td><td>${pill(m.bands.air)}</td><td>${m.originalLufs.toFixed(1)}</td><td>${m.lagMs>0?'+':''}${m.lagMs.toFixed(1)} ms</td><td>${m.syncSpreadMs.toFixed(1)} ms</td></tr>`).join('')}
function relativeCurve(i){const c=data.microphones[i].curve,r=data.microphones[reference].curve;const out=c.map((v,j)=>v-r[j]);const f=data.frequencies,mid=out.filter((_,j)=>f[j]>=500&&f[j]<=2000).sort((a,b)=>a-b);const zero=mid[Math.floor(mid.length/2)]||0;return out.map(v=>v-zero)}
function supportRange(coverage){const indices=coverage.map((v,i)=>v>=.4?i:-1).filter(i=>i>=0);if(!indices.length)return '뚜렷한 대역 없음';const low=data.frequencies[indices[0]],high=data.frequencies[indices.at(-1)];const unit=f=>f>=1000?`${(f/1000).toFixed(1)} kHz`:`${Math.round(f)} Hz`;return `${unit(low)}–${unit(high)}`}
function updateCoverageReadout(){const chosen=hoveredMic===null?null:data.microphones[hoveredMic];$('coverage-readout').textContent=`${chosen?chosen.label:'전체 녹음'} · 신호가 비교적 풍부한 대역 ${supportRange(chosen?.coverage||data.coverage)}`}
function draw(){
  if(!data)return;
  const canvas=$('chart'),rect=canvas.getBoundingClientRect(),dpr=devicePixelRatio||1;
  canvas.width=rect.width*dpr;canvas.height=rect.height*dpr;
  const g=canvas.getContext('2d');g.scale(dpr,dpr);
  const W=rect.width,H=rect.height,L=49,R=14,T=16,B=37,PW=W-L-R,PH=H-T-B;
  const focused=$('focus-band').checked,minF=focused?70:40,maxF=focused?4000:18000;
  const x=f=>L+(Math.log(f)-Math.log(minF))/(Math.log(maxF)-Math.log(minF))*PW;
  const y=v=>T+(18-v)/36*PH;
  const visible=data.frequencies.map((f,i)=>f>=minF&&f<=maxF?i:-1).filter(i=>i>=0);
  g.clearRect(0,0,W,H);
  // Shared programme support forms a soft frequency-dependent floor, not a
  // confidence gradient on the lines themselves.
  const floor=g.createLinearGradient(L,0,W-R,0);
  const floorColor=i=>`rgba(213,243,106,${(.03+.28*data.coverage[i]).toFixed(3)})`;
  floor.addColorStop(0,floorColor(visible[0]));
  visible.forEach(j=>floor.addColorStop(Math.max(0,Math.min(1,(x(data.frequencies[j])-L)/PW)),floorColor(j)));
  floor.addColorStop(1,floorColor(visible.at(-1)));
  g.fillStyle=floor;g.fillRect(L,H-B-34,PW,34);
  g.font='10px DM Mono, monospace';g.textAlign='right';
  for(const db of [-15,-10,-5,0,5,10,15]){const yy=y(db);g.strokeStyle=db===0?'#637077':'#273644';g.lineWidth=db===0?1.4:1;g.beginPath();g.moveTo(L,yy);g.lineTo(W-R,yy);g.stroke();g.fillStyle='#80909b';g.fillText(`${db>0?'+':''}${db}`,L-8,yy+3)}
  for(const f of (focused?[70,100,200,500,1000,2000,4000]:[50,100,200,500,1000,2000,5000,10000])){const xx=x(f);g.strokeStyle='#26333e';g.lineWidth=1;g.beginPath();g.moveTo(xx,T);g.lineTo(xx,H-B);g.stroke();g.fillStyle='#80909b';g.textAlign='center';g.fillText(f>=1000?`${f/1000}k`:`${f}`,xx,H-14)}
  if(hoveredMic!==null){
    const i=hoveredMic,c=relativeCurve(i),coverage=data.microphones[i].coverage,rgb=colors[i].match(/[a-f0-9]{2}/gi).map(v=>parseInt(v,16));
    // Each narrow trapezoid is vertically faded and its opacity is weighted
    // by programme support at that frequency. The curve stroke stays solid.
    for(let j=0;j<c.length-1;j++){
      if(data.frequencies[j]<minF||data.frequencies[j+1]>maxF)continue;
      const weight=(coverage[j]+coverage[j+1])/2;if(weight<.005)continue;
      const x1=x(data.frequencies[j]),x2=x(data.frequencies[j+1]),y1=y(Math.max(-18,Math.min(18,c[j]))),y2=y(Math.max(-18,Math.min(18,c[j+1])));
      const grad=g.createLinearGradient(0,Math.min(y1,y2),0,H-B);
      grad.addColorStop(0,`rgba(${rgb.join(',')},${(.12+.34*weight).toFixed(3)})`);
      grad.addColorStop(1,`rgba(${rgb.join(',')},${(.02+.08*weight).toFixed(3)})`);
      g.fillStyle=grad;g.beginPath();g.moveTo(x1,y1);g.lineTo(x2,y2);g.lineTo(x2,H-B);g.lineTo(x1,H-B);g.closePath();g.fill();
    }
  }
  data.microphones.forEach((m,i)=>{if(hoveredMic===null?(!selected.has(i)||i===reference):i!==hoveredMic)return;const c=relativeCurve(i);g.beginPath();visible.forEach((j,k)=>{const xx=x(data.frequencies[j]),yy=y(Math.max(-18,Math.min(18,c[j])));k?g.lineTo(xx,yy):g.moveTo(xx,yy)});g.strokeStyle=colors[i];g.lineWidth=hoveredMic===i?3.2:2.2;g.lineJoin='round';g.stroke()});
  updateCoverageReadout();
}
function legend(){$('legend').replaceChildren(...data.microphones.map((m,i)=>{const label=document.createElement('label');label.className='legend-item';label.tabIndex=0;label.innerHTML=`<input type="checkbox" ${selected.has(i)?'checked':''} ${i===reference?'disabled':''}><span class="swatch" style="background:${colors[i]}"></span>${m.label}${i===reference?' (기준)':''}`;label.querySelector('input').addEventListener('change',e=>{e.target.checked?selected.add(i):selected.delete(i);draw()});const enter=()=>{hoveredMic=i;label.classList.add('is-hovered');draw()},leave=()=>{hoveredMic=null;label.classList.remove('is-hovered');draw()};label.addEventListener('mouseenter',enter);label.addEventListener('mouseleave',leave);label.addEventListener('focusin',enter);label.addEventListener('focusout',leave);return label}))}
async function init(){
  data=await fetch('./data/analysis.json?v=iphone2km184-iir').then(r=>r.json());regionEnd=data.durationSec;
  $('duration').textContent=`${data.durationSec.toFixed(1)}s`;$('target').textContent=`${data.targetLufs.toFixed(1)}`;$('total').textContent=fmt(data.durationSec);$('waveform-total').textContent=fmt(data.durationSec);
  data.microphones.forEach((m,i)=>{const o=document.createElement('option');o.value=i;o.textContent=m.label;$('reference').append(o);if(m.id===data.reference)reference=i});
  active=reference;$('reference').value=reference;selected=new Set(data.microphones.map((_,i)=>i).filter(i=>i!==reference));drawTable();legend();renderTracks();updateRegion();draw();requestAnimationFrame(tick);
  context=new (window.AudioContext||window.webkitAudioContext)();gain=context.createGain();gain.gain.value=+$('volume').value/100;gain.connect(context.destination);
  $('play').disabled=true;$('replay-region').disabled=true;$('audio-status').textContent=`${data.microphones.length}개 트랙을 불러오는 중…`;
  try{const response=await fetch('./data/waveform.json');if(response.ok){waveformEnvelope=await response.json();drawWaveform()}}catch(e){console.warn('파형 데이터 로딩 실패, 음원에서 생성합니다',e)}
  buffers=await Promise.all(data.microphones.map(async m=>{const response=await fetch(encodeURI(m.audio));if(!response.ok)throw Error(`${m.label} 음원 로딩 실패`);return context.decodeAudioData(await response.arrayBuffer())}));
  if(!waveformEnvelope.length)buildWaveform(buffers[reference]);
  $('play').disabled=false;$('replay-region').disabled=false;$('audio-status').textContent='준비 완료 · Solo 버튼으로 트랙을 바꾸세요';
}
$('waveform').addEventListener('pointerdown',e=>{if(!waveformEnvelope.length)return;e.preventDefault();$('waveform').setPointerCapture(e.pointerId);waveformDrag={start:waveformTime(e.clientX),startX:e.clientX,moved:false};});
$('waveform').addEventListener('pointermove',e=>{if(!waveformDrag)return;const t=waveformTime(e.clientX);if(Math.abs(e.clientX-waveformDrag.startX)>5)waveformDrag.moved=true;if(waveformDrag.moved){const a=Math.min(waveformDrag.start,t),b=Math.max(waveformDrag.start,t);$('waveform-region').style.left=regionPct(a);$('waveform-region').style.width=regionPct(b-a)}});
$('waveform').addEventListener('pointerup',e=>{if(!waveformDrag)return;const t=waveformTime(e.clientX);if(waveformDrag.moved&&Math.abs(t-waveformDrag.start)>=.25){const a=Math.min(waveformDrag.start,t),b=Math.max(waveformDrag.start,t);setWaveformRegion(a,b)}else seekTo(t);waveformDrag=null;syncTimeline()});
$('waveform').addEventListener('pointercancel',()=>{waveformDrag=null;syncTimeline()});
$('play').addEventListener('click',()=>playing?pause():play());$('seek').addEventListener('input',e=>seekTo(e.target.value/1000*data.durationSec));$('volume').addEventListener('input',e=>{if(gain)gain.gain.value=e.target.value/100});$('region-start').addEventListener('input',()=>updateRegion('start'));$('region-end').addEventListener('input',()=>updateRegion('end'));$('set-start').addEventListener('click',()=>{$('region-start').value=Math.round(time()/data.durationSec*1000);updateRegion('start')});$('set-end').addEventListener('click',()=>{$('region-end').value=Math.round(time()/data.durationSec*1000);updateRegion('end')});$('reset-region').addEventListener('click',()=>{$('region-start').value=0;$('region-end').value=1000;updateRegion()});$('replay-region').addEventListener('click',playRegion);$('loop').addEventListener('change',()=>{if(playing){position=time();playing=false;stopSource();startSource()}});$('focus-band').addEventListener('change',draw);$('reference').addEventListener('change',e=>{reference=+e.target.value;selected.delete(reference);legend();draw()});$('select-all').addEventListener('click',()=>{selected=new Set(data.microphones.map((_,i)=>i).filter(i=>i!==reference));legend();draw()});$('clear').addEventListener('click',()=>{selected.clear();legend();draw()});window.addEventListener('resize',()=>{draw();drawWaveform()});init().catch(e=>{console.error(e);$('audio-status').textContent='음원을 불러오지 못했습니다. 페이지를 새로고침해 주세요.'});
