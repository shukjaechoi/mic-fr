"""Render synchronized listening page for the causal attack-controller trial."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy import signal, ndimage
from PIL import Image
from matplotlib import colormaps

ROOT=Path(__file__).resolve().parents[2]
PUB=ROOT/'docs/experiments/20261008-iphone-km184-public'
data=json.loads((PUB/'attack-controller-trial.json').read_text())
FS=48000; NF=240; NT=470
freq=np.geomspace(6000,80,NF)
variants=('km184','raw','shaped','dynamic','neural')
names={'km184':'KM184 기준','raw':'기기 원본','shaped':'기존 고정 EQ','dynamic':'어택 동적 EQ −2dB','neural':'작은 뉴럴 제어기'}
for c in data['clips']:
    spectra={}
    for k in variants:
        y,_=sf.read(PUB/c['media'][k]['audio'])
        f,t,z=signal.stft(y,FS,nperseg=2048,noverlap=1536,boundary=None)
        db=20*np.log10(np.maximum(abs(z),1e-8))
        fi,ti=np.meshgrid(freq/(f[1]-f[0]),(np.linspace(0,5,NT)-t[0])/(t[1]-t[0]),indexing='ij')
        spectra[k]=ndimage.map_coordinates(db,[fi,ti],order=1,mode='nearest')
    top=max(v.max() for v in spectra.values())
    rs=ndimage.gaussian_filter(spectra['km184'],sigma=(2,1.5))
    for k in variants:
        stem=f'attack-{c["group"]}-{c["start_s"]}-{c["device"]}-{k}'
        rgb=colormaps['magma'](np.clip((spectra[k]-(top-72))/72,0,1))[:,:,:3]
        Image.fromarray(np.uint8(rgb*255),'RGB').save(PUB/(stem+'.png'),optimize=True)
        c['media'][k]['spectrogram']=stem+'.png'
        if k!='km184':
            diff=np.clip(ndimage.gaussian_filter(spectra[k],sigma=(2,1.5))-rs,-12,12)
            observed=(rs>top-58)&(spectra[k]>top-58)&(freq[:,None]>=100)
            rgb=colormaps['RdBu_r']((diff+12)/24)[:,:,:3]
            rgb[~observed]=[.88,.90,.91]
            Image.fromarray(np.uint8(rgb*255),'RGB').save(PUB/(stem+'-error.png'),optimize=True)
            c['media'][k]['residual']=stem+'-error.png'
    print('images',c['device'],c['group'],flush=True)
(PUB/'attack-controller-trial.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')

inline=json.dumps({'clips':data['clips'],'bands_hz':data['bands_hz']},ensure_ascii=False,separators=(',',':'))
html='''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>퍼커션 어택 보완 필터 비교</title>
<style>:root{font-family:system-ui,-apple-system,sans-serif;color:#17303a;background:#f4f8f8}body{margin:0}main{max-width:1260px;margin:auto;padding:22px}a{color:#087c87}nav{display:flex;flex-wrap:wrap;gap:14px}h1{font-size:clamp(1.5rem,3vw,2.3rem)}p{line-height:1.55}.lead{background:#e5f2f1;border-left:5px solid #087c87;padding:15px;border-radius:8px}.layout{display:grid;grid-template-columns:190px minmax(0,1fr);gap:16px}.side{display:grid;align-content:start;gap:8px;position:sticky;top:15px}.dev{display:flex;gap:8px}button{font:inherit;background:white;border:1px solid #b9cbd0;border-radius:7px;padding:10px;cursor:pointer}button[aria-selected=true]{background:#087c87;color:white}.transport{position:sticky;top:0;z-index:3;background:#e5eff0;padding:9px;display:flex;gap:8px;align-items:center;border-radius:8px}.transport input{flex:1}.card{background:white;border:1px solid #d2e0e3;border-radius:9px;padding:12px;margin:12px 0}.card h3{margin:0 0 8px}.images{display:grid;grid-template-columns:1fr 1fr;gap:7px}.image{position:relative;cursor:pointer;line-height:0}.image img{width:100%;display:block;aspect-ratio:470/240}.cursor{position:absolute;top:0;bottom:0;left:0;width:2px;background:#00ffff;pointer-events:none}.axis{display:flex;justify-content:space-between;font-size:.8rem;color:#567}table{border-collapse:collapse;background:white}td,th{border:1px solid #cedbde;padding:8px}th{background:#e8f1f3}.small{font-size:.9rem;color:#516b74}@media(max-width:800px){.layout{grid-template-columns:1fr}.side{display:flex;position:static;overflow:auto}.images{grid-template-columns:1fr}}</style>
<main><nav><a href="upper-mid-supplement.html">기존 고정 EQ 비교</a><a href="october-four-device-eq.html">네 기기 비교</a><a href="index.html">홈</a></nav><h1>어택 순간의 상위 대역 보완 시험</h1>
<p class="lead">기존 고정 EQ 뒤에 iPhone은 1.6–3.2kHz, S21은 3.2–6kHz를 <b>타격 직후 약 50ms 동안 최대 2dB</b> 낮추는 후보를 적용했습니다. 단순 동적 EQ와 작은 뉴럴 제어기를 같은 시간·같은 크기 제한으로 비교합니다. 아래 음원은 실제 필터 출력입니다.</p>
<p>단순 방식은 5ms 신호 변화가 6dB를 넘는 어택을 감지해 해당 대역을 2dB 줄입니다. 뉴럴 방식은 같은 어택 검출 후 여섯 가지 현재·직전 대역 에너지에서 감쇄량을 정합니다. 퍼커션과 복합화음은 뉴럴 학습에서 제외했습니다. 단, 앞단의 기존 EQ는 10월 자료로 만든 것이므로 독립 검증은 아닙니다. 두 방식 모두 현재 5ms 블록의 정보를 쓰므로 처리 시 약 5ms 버퍼가 필요합니다.</p>
<div class="layout"><div id="groups" class="side"></div><section><div id="devices" class="dev"></div><p id="selected"></p><div class="transport"><button id="toggle">재생 / 일시정지</button><input id="seek" type="range" min="0" max="5" step="0.01" value="0" aria-label="공통 재생 위치"><span id="clock">0.0 / 5.0초</span></div><div id="cards"></div></section></div>
<h2>퍼커션 타격 수치</h2><p>타격 직후 60ms의 높은 대역 / 0.8–1.6kHz 비율이 KM184보다 얼마나 큰지 표시합니다. 양수는 해당 대역이 상대적으로 강함을 뜻합니다. 두 타격만을 묘사하므로 다른 퍼커션 전체에 일반화하지 않습니다.</p><div style="overflow:auto"><table><thead><tr><th>기기·타격</th><th>원본</th><th>기존 EQ</th><th>단순 동적 EQ</th><th>뉴럴 제어</th></tr></thead><tbody id="metrics"></tbody></table></div>
<p class="small">이번 −2dB 제한에서는 iPhone의 두 타격에서 단순 방식이 약 0.7–0.9dB, S21에서는 약 0.3–0.4dB의 상대 차이를 줄였습니다. 작은 뉴럴 모델이 단순 방식보다 분명히 낫다는 근거는 아직 없습니다. S21의 큰 잔차는 여전히 남습니다. 현재 출력은 청취 비교용 동일 기준 레벨이며, 다른 녹음에서의 최적 보정이나 실시간 앱의 지연·안정성은 검증하지 않았습니다. <a href="attack-controller-trial.json">필터·뉴럴 계수와 수치</a></p></main>
<script>const D=__DATA__,keys=['km184','raw','shaped','dynamic','neural'],names={km184:'KM184 기준',raw:'기기 원본',shaped:'기존 고정 EQ',dynamic:'어택 동적 EQ −2dB',neural:'작은 뉴럴 제어기'};let group='percussion',device='iphone',selected='dynamic',audio=null,pos=0;const A={};function item(){return D.clips.find(c=>c.group===group&&c.device===device)}function stop(){if(audio){audio.pause();pos=audio.currentTime||pos;audio=null}}function update(){if(audio&&!audio.paused)pos=audio.currentTime;document.getElementById('seek').value=pos;document.getElementById('clock').textContent=pos.toFixed(1)+' / 5.0초';document.querySelectorAll('.cursor').forEach(x=>x.style.left=pos/5*100+'%');if(audio&&!audio.paused)requestAnimationFrame(update)}function play(k,p){let same=audio===A[k]&&!audio.paused;stop();selected=k;if(p!==undefined)pos=p;if(same&&p===undefined){update();return}audio=A[k];audio.currentTime=Math.min(pos,4.98);audio.play().catch(()=>{});update()}function render(reset=true){stop();if(reset)pos=0;let c=item();document.querySelectorAll('#groups button').forEach(b=>b.setAttribute('aria-selected',b.dataset.group===group));document.querySelectorAll('#devices button').forEach(b=>b.setAttribute('aria-selected',b.dataset.device===device));document.getElementById('selected').textContent=c.group+' · KM184 '+c.start_s+'–'+(c.start_s+5)+'초 · '+device.toUpperCase()+' · 모든 음원 재생 위치 동기화';let box=document.getElementById('cards');box.replaceChildren();for(let k of keys){let m=c.media[k],card=document.createElement('div');card.className='card';card.innerHTML=`<h3>${names[k]}</h3><div class="images"><div class="image" data-key="${k}"><img src="${m.spectrogram}" alt="${names[k]} 스펙트로그램"><span class="cursor"></span></div><div class="image" data-key="${k}">${m.residual?`<img src="${m.residual}" alt="KM184와의 잔차">`:'<div style="padding:20px">KM184 기준</div>'}<span class="cursor"></span></div></div><div class="axis"><span>0초</span><span>1초</span><span>2초</span><span>3초</span><span>4초</span><span>5초</span></div><button data-play="${k}">${names[k]} 재생</button>`;box.append(card);A[k]=new Audio(m.audio);A[k].onended=()=>{audio=null;pos=5;update()}}box.querySelectorAll('[data-play]').forEach(b=>b.onclick=()=>play(b.dataset.play));box.querySelectorAll('.image').forEach(el=>el.onclick=e=>{let r=el.getBoundingClientRect();play(el.dataset.key,Math.max(0,Math.min(4.98,(e.clientX-r.left)/r.width*5)))});update()}document.getElementById('groups').innerHTML=[['percussion','퍼커션'],['open_notes','개방현 단음'],['complex','복합화음']].map(([g,n])=>`<button data-group="${g}">${n}</button>`).join('');document.querySelectorAll('#groups button').forEach(b=>b.onclick=()=>{group=b.dataset.group;render()});document.getElementById('devices').innerHTML=['iphone','s21'].map(d=>`<button data-device="${d}">${d.toUpperCase()}</button>`).join('');document.querySelectorAll('#devices button').forEach(b=>b.onclick=()=>{device=b.dataset.device;render(false)});document.getElementById('seek').oninput=e=>{stop();pos=+e.target.value;update()};document.getElementById('toggle').onclick=()=>play(selected);document.getElementById('metrics').innerHTML=D.clips.filter(c=>c.group==='percussion').flatMap(c=>c.attack_excess_db.map((v,i)=>`<tr><td>${c.device.toUpperCase()} · ${i+1}타격</td>${['raw','shaped','dynamic','neural'].map(k=>`<td>${v[k].toFixed(2)}dB</td>`).join('')}</tr>`)).join('');render();</script></html>'''.replace('__DATA__',inline)
(PUB/'attack-control.html').write_text(html)
print(PUB/'attack-control.html')
