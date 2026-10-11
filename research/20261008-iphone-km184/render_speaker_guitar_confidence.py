"""Publish synchronized evidence and listening page for the confidence EQ trial."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy import signal, ndimage
from PIL import Image
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / 'docs/experiments/20261008-iphone-km184'
PUBLIC = ROOT / 'docs/experiments/20261008-iphone-km184-public'
data = json.loads((LOCAL / 'speaker-guitar-confidence-eq.json').read_text())
old = json.loads((ROOT / 'docs/experiments/20261002/data/analysis.json').read_text())
maps = {m['id']: m['guitarSync'] for m in old['microphones']}
maps['iphone'] = json.loads((LOCAL / 'analysis.json').read_text())['sets']['guitar']['mapping']
audio = {d: np.load(f'/private/tmp/mic-audit-20261008/{d}-guitar.npy', mmap_mode='r') for d in ('km184', 'iphone', 's21')}
FS = 48000
clips = [('arpeggio', 100, '아르페지오'), ('open_notes', 418, '개방현 단음'), ('complex', 1704, '복합 화음'), ('percussion', 1792, '퍼커션')]
freqgrid = np.geomspace(6000, 80, 240)
def spec(x):
    f, t, z = signal.stft(x, FS, nperseg=2048, noverlap=1536, boundary=None)
    db = 20 * np.log10(np.maximum(abs(z), 1e-8))
    fi, ti = np.meshgrid(freqgrid / (f[1] - f[0]), (np.linspace(0, 5, 470) - t[0]) / (t[1] - t[0]), indexing='ij')
    return ndimage.map_coordinates(db, [fi, ti], order=1, mode='nearest')
def mapped(m, t): return round(t * FS * m['slope'] + m.get('interceptSamples', m.get('intercept_samples', 0)))
def saveimg(path, array): Image.fromarray(np.uint8(np.clip(array * 255, 0, 255)), 'RGB').save(PUBLIC / path, optimize=True)

manifest = []
for group, t, label in clips:
    ref = np.asarray(audio['km184'][t * FS:(t + 5) * FS], float)
    for d in ('iphone', 's21'):
        start = mapped(maps[d], t); left = max(0, start - FS // 4); off = start - left
        x = np.asarray(audio[d][left:start + 5 * FS], float)
        raw = x[off:]; existing = signal.sosfilt(np.asarray(json.loads((PUBLIC / 'upper-mid-supplement-browser.json').read_text())['models'][d]['sos']['shaped']), x)[off:]
        new = signal.sosfilt(np.asarray(data['models'][d]['sos']), x)[off:]
        f, rp = signal.welch(ref, FS, nperseg=8192, noverlap=4096)
        _, xp = signal.welch(raw, FS, nperseg=8192, noverlap=4096)
        def am(c): return (f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))
        anchor = np.median([10 * np.log10(np.mean(rp[am(c)]) / np.mean(xp[am(c)])) for c in (335.7, 597.7)])
        gain = 10 ** (anchor / 20)
        waves = {'km184': ref, 'raw': raw * gain, 'existing': existing * gain, 'new': new * gain}
        safety = min(1, .8 / max(float(np.max(abs(v))) for v in waves.values()))
        waves = {k: v * safety for k, v in waves.items()}
        specs = {k: spec(v) for k, v in waves.items()}; top = max(float(v.max()) for v in specs.values())
        refspec = ndimage.gaussian_filter(specs['km184'], (2, 1.5)); media = {}
        for key, wave in waves.items():
            stem = f'speaker-guitar-{group}-{t}-{d}-{key}'
            sf.write(PUBLIC / (stem + '.wav'), wave, FS, subtype='PCM_16')
            rgb = colormaps['magma'](np.clip((specs[key] - top + 72) / 72, 0, 1))[:, :, :3]
            for sec in range(1, 5): rgb[:, round(sec / 5 * 469), :] = rgb[:, round(sec / 5 * 469), :] * .78 + .22
            saveimg(stem + '.png', rgb)
            item = {'audio': stem + '.wav', 'spectrogram': stem + '.png'}
            if key != 'km184':
                diff = np.clip(ndimage.gaussian_filter(specs[key], (2, 1.5)) - refspec, -12, 12)
                observed = (refspec > top - 58) & (specs[key] > top - 58) & (freqgrid[:, None] >= 100)
                colors = colormaps['RdBu_r']((diff + 12) / 24)[:, :, :3]
                colors[~observed] = [.88, .90, .91]
                saveimg(stem + '-error.png', colors)
                item['residual'] = stem + '-error.png'
            media[key] = item
        manifest.append({'group': group, 'label': label, 'start_s': t, 'device': d, 'media': media, 'anchor_db': round(float(anchor), 3)})
        print(group, d, flush=True)

payload = {'analysis': data, 'clips': manifest}
(PUBLIC / 'speaker-guitar-confidence-eq-browser.json').write_text(json.dumps(payload, ensure_ascii=False, separators=(',', ':')) + '\n')

html = r'''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>스피커·기타 신뢰도 기반 중고역 점검</title>
<style>:root{font-family:system-ui,-apple-system,sans-serif;color:#18323c;background:#f4f8f8}body{margin:0}main{max-width:1250px;margin:auto;padding:20px}a{color:#087e8b}nav{display:flex;gap:14px;flex-wrap:wrap}.lead{background:#e7f4f1;padding:15px;border-left:5px solid #09877f;border-radius:8px;line-height:1.55}.warn{background:#fff2e7;border-left-color:#e2873a}h1{font-size:clamp(1.6rem,3vw,2.3rem)}h2{margin-top:30px}p{line-height:1.55}.layout{display:grid;grid-template-columns:190px minmax(0,1fr);gap:16px}.side{display:grid;align-content:start;gap:8px;position:sticky;top:10px}button{font:inherit;padding:9px;border:1px solid #bcd0d2;border-radius:7px;background:#fff;color:#25434d;cursor:pointer}button[aria-selected=true]{background:#087e8b;color:white}.devices{display:flex;gap:8px;margin-bottom:10px}.panel{background:white;border:1px solid #d3e0e3;border-radius:8px;padding:12px;margin:12px 0;overflow:auto}.transport{position:sticky;top:0;background:#e7f3f4;z-index:4;padding:9px;border-radius:8px;display:flex;gap:10px;align-items:center}.transport input{flex:1}.specrow{display:grid;grid-template-columns:1fr 1fr;gap:8px}.spec{position:relative;cursor:pointer}.spec img{width:100%;display:block;aspect-ratio:470/240}.cursor{position:absolute;left:0;top:0;bottom:0;background:#3cf1ef;width:2px;pointer-events:none}.axis{display:flex;justify-content:space-between;color:#6a7d83;font-size:.83rem}.play{margin-top:8px}table{width:100%;border-collapse:collapse;min-width:650px}td,th{border-bottom:1px solid #e0e9eb;padding:8px;text-align:left}th{background:#e9f1f2}svg{width:100%;min-width:620px}.muted{color:#57717a;font-size:.92rem}@media(max-width:800px){.layout{grid-template-columns:1fr}.side{position:static;display:flex;overflow:auto}.side button{min-width:140px}.specrow{grid-template-columns:1fr}}</style>
<main><nav><a href="index.html">실험 홈</a><a href="upper-mid-supplement.html">이전 보완 필터</a><a href="joint-event-timbre-qc.html">사건별 음색 근거</a><a href="temporal-spread-qc.html">시간 분석</a><a href="brightness-extension.html">남은 밝기 · 6–10kHz 보완</a></nav><h1>스피커 기준 + 주법별 신뢰도로 다시 만든 중고역 후보</h1>
<p class="lead">핑크·화이트 노이즈와 로그 스윕의 공통 방향을 <b>스피커 기준의 잠정 힌트</b>로 쓰고, 기타에서 해당 대역이 양쪽 녹음에 충분히 나타나는 주법만 가중했습니다. 기존 필터의 중고역을 무조건 더 세게 누르지 않고, 넓은 두 단의 EQ로 시험했습니다. 스피커와 기타의 배치가 다르므로 이 결과는 직접음 역필터가 아닙니다.</p>
<p class="lead warn"><b>결과:</b> S21은 3–5kHz의 반복된 과다를 완만하게 낮추는 후보가 기존보다 개선됩니다. iPhone은 스피커 기준을 더 반영한 후보가 기타의 넓은 대역 오차를 오히려 키웁니다. 따라서 <b>iPhone 새 후보는 채택하지 않습니다.</b> 아래 청취는 원인을 확인하는 비교이며, 더 비슷하게 들린다고 미리 주장하지 않습니다.</p>
<h2>대역 근거</h2><p>양수는 KM184보다 기기가 약해 부스트 방향, 음수는 기기가 강해 컷 방향입니다. 기타 신뢰도는 양쪽의 배경 대비 신호와 각 주법에서 해당 대역의 에너지 비중으로 정했습니다. 스피커 측 세 신호의 차이가 크거나 기타에서 보정 방향이 반대면 필터 적합 가중치를 낮췄습니다. 선형 스윕은 이 구간의 전체 PSD가 다른 세 신호와 크게 어긋나 설계에서 제외하고 QC 값만 저장했습니다.</p><div class="panel"><table><thead><tr><th>기기·대역</th><th>핑크</th><th>화이트</th><th>로그</th><th>기타 주법 지지</th><th>적합 가중치</th></tr></thead><tbody id="evidence"></tbody></table></div>
<h2>실제 필터 응답</h2><p>iPhone은 기존 필터 위에 새 후보 두 단을 더한 진단용 비교입니다. S21은 기존 저역 세 단을 그대로 두고 중고역 부분을 새로 구성했습니다.</p><div class="panel"><svg id="response" viewBox="0 0 860 260" role="img" aria-label="기기별 기존 및 새 필터의 주파수 응답"></svg></div>
<h2>주법별 수치</h2><p>5초 구간의 관측 가능한 0.8–5.08kHz 대역에서 KM184와의 평균 절대 음색 오차(dB)입니다. 각 대역의 배경 대비 신호와 그 주법의 에너지 비중으로 가중했고, 같은 구간·동일 기준 레벨을 보정 전후에 적용했습니다. 이 수치는 질감 전체나 청감 점수가 아닙니다.</p><div class="panel"><table><thead><tr><th>기기·주법</th><th>유효 창</th><th>원본</th><th>기존 필터</th><th>새 후보</th><th>새 후보 악화 창</th></tr></thead><tbody id="scores"></tbody></table></div>
<h2>같은 시각에서 듣고 보기</h2><p>왼쪽 주법과 기기를 선택하세요. 재생 위치를 옮기거나 스펙트로그램을 누르면 네 버전의 같은 순간으로 이동합니다. 기준·원본·기존·새 후보의 스펙트로그램은 같은 밝기 범위이며, 오른쪽 잔차의 회색 부분은 해당 순간 비교가 불안정한 곳입니다.</p>
<div class="layout"><div class="side" id="groups"></div><div><div class="devices" id="devices"></div><p id="selected"></p><div class="transport"><button id="toggle">재생 / 정지</button><input id="seek" type="range" min="0" max="5" step="0.01" value="0" aria-label="공통 재생 위치"><span id="time">0.0 / 5.0초</span></div><div id="versions"></div></div></div>
<p class="muted">동일 10월 자료에서 설계·평가한 탐색 결과입니다. 주법별 점수도 완전한 독립 평가가 아닙니다. 퍼커션 S21 유효 창은 2개뿐입니다. 전체 파일을 연속 처리한 출력 감쇠 전 샘플 피크: iPhone 새 후보 −0.80dBFS, S21 새 후보 +2.96dBFS. S21은 기존과 같은 −6.5dB 출력 감쇠를 적용하면 약 −3.54dBFS입니다. 이는 샘플 피크이며 true peak나 실제 앱의 입력 여유를 검증한 값은 아닙니다. 청취 클립은 버전 간 공통 표시 배율을 따로 적용했습니다. 소스와 계산 기록: <a href="speaker-guitar-confidence-eq-browser.json">계수·신뢰도·창별 점수 JSON</a>.</p></main>
<script>const p=__DATA__,a=p.analysis;const names={arpeggio:'아르페지오',open_notes:'개방현 단음',complex:'복합 화음',percussion:'퍼커션',strum:'스트로크',scale:'스케일',dynamics:'다이내믹',slur:'슬러',glissando:'글리산도',tremolo:'트레몰로',rasgueado:'라스게아도'};const labels={km184:'KM184 기준',raw:'기기 원본',existing:'기존 필터',new:'스피커·기타 신뢰도 후보'};let group='complex',device='iphone',which='new',audio=null,pos=0;let players={};const seek=document.getElementById('seek');function clip(){return p.clips.find(x=>x.group===group&&x.device===device)}function stop(){if(audio){audio.pause();pos=audio.currentTime||pos;audio=null}}function sync(){if(audio&&!audio.paused)pos=audio.currentTime;seek.value=pos;document.getElementById('time').textContent=pos.toFixed(1)+' / 5.0초';document.querySelectorAll('.cursor').forEach(x=>x.style.left=(pos*20)+'%');if(audio&&!audio.paused)requestAnimationFrame(sync)}function play(k,at){let same=audio===players[k]&&!audio.paused;stop();which=k;if(at!==undefined)pos=at;if(same&&at===undefined){sync();return}let x=players[k];x.currentTime=Math.min(4.98,pos);audio=x;x.play().catch(()=>{});sync()}function render(){stop();pos=0;let c=clip();document.querySelectorAll('#groups button').forEach(x=>x.setAttribute('aria-selected',x.dataset.group===group));document.querySelectorAll('#devices button').forEach(x=>x.setAttribute('aria-selected',x.dataset.device===device));document.getElementById('selected').textContent=c.label+' · KM184 '+c.start_s+'–'+(c.start_s+5)+'초 · '+device.toUpperCase();let box=document.getElementById('versions');box.replaceChildren();players={};for(let k of ['km184','raw','existing','new']){let m=c.media[k],card=document.createElement('div');card.className='panel';card.innerHTML='<b>'+labels[k]+'</b><div class="specrow"><div class="spec" data-key="'+k+'"><img src="'+m.spectrogram+'" alt="'+labels[k]+' 스펙트로그램"><span class="cursor"></span></div><div class="spec" data-key="'+k+'">'+(m.residual?'<img src="'+m.residual+'" alt="KM184 대비 잔차">':'<p>기준 녹음</p>')+'<span class="cursor"></span></div></div><div class="axis"><span>0초</span><span>1초</span><span>2초</span><span>3초</span><span>4초</span><span>5초</span></div><button class="play" data-play="'+k+'">'+labels[k]+' 재생</button>';box.append(card);players[k]=new Audio(m.audio);players[k].onended=()=>{audio=null;pos=5;sync()}}box.querySelectorAll('[data-play]').forEach(x=>x.onclick=()=>play(x.dataset.play));box.querySelectorAll('.spec').forEach(x=>x.onclick=e=>{let r=x.getBoundingClientRect();play(x.dataset.key,Math.max(0,Math.min(4.98,(e.clientX-r.left)/r.width*5)))});graph();sync()}function graph(){let s='<rect width="860" height="260" fill="white"/>';let fs=a.response_hz,x=f=>60+Math.log(f/100)/Math.log(8000/100)*740,y=v=>125-v*13;for(let v of [-8,-4,0,4,8])s+='<line x1="60" x2="800" y1="'+y(v)+'" y2="'+y(v)+'" stroke="'+(v===0?'#8ca0a4':'#e2e9eb')+'"/><text x="15" y="'+(y(v)+4)+'" font-size="12">'+v+'</text>';for(let f of [100,300,1000,2000,4000,8000])s+='<text x="'+x(f)+'" y="246" text-anchor="middle" font-size="12">'+(f>=1000?f/1000+'k':f)+'</text>';for(let [k,color] of [['existing','#6e8793'],['new','#0a8582']])s+='<polyline points="'+fs.map((f,i)=>x(f)+','+y(a.response_db[device][k][i])).join(' ')+'" fill="none" stroke="'+color+'" stroke-width="3"/>';s+='<text x="135" y="20" fill="#6e8793">기존</text><text x="245" y="20" fill="#0a8582">새 후보</text>';document.getElementById('response').innerHTML=s}function tables(){let h='';for(let d of ['iphone','s21']){let e=a.evidence[d];for(let i=0;i<a.centers_hz.length;i++)h+='<tr><td>'+d.toUpperCase()+' · '+Math.round(a.centers_hz[i])+'Hz</td>'+['pink','white','log'].map(s=>'<td>'+e.speaker_km_minus_device_db[s][i].toFixed(1)+'</td>').join('')+'<td>'+e.guitar_support_groups[i]+'/11'+(e.same_direction[i]?' · 같은 방향':' · 방향 다름')+'</td><td>'+e.fit_weight[i].toFixed(2)+'</td></tr>'}document.getElementById('evidence').innerHTML=h;h='';for(let d of ['iphone','s21'])for(let [g,r] of Object.entries(a.summary[d].groups))h+='<tr><td>'+d.toUpperCase()+' · '+(names[g]||g)+'</td><td>'+r.windows+'</td><td>'+r.mean_mae_db.raw.toFixed(2)+'</td><td>'+r.mean_mae_db.existing.toFixed(2)+'</td><td><b>'+r.mean_mae_db.new.toFixed(2)+'</b></td><td>'+r.new_worse_than_existing_gt_0_5db+'</td></tr>';document.getElementById('scores').innerHTML=h}document.getElementById('groups').innerHTML=['arpeggio','open_notes','complex','percussion'].map(g=>'<button data-group="'+g+'">'+names[g]+'</button>').join('');document.querySelectorAll('#groups button').forEach(x=>x.onclick=()=>{group=x.dataset.group;render()});document.getElementById('devices').innerHTML=['iphone','s21'].map(d=>'<button data-device="'+d+'">'+d.toUpperCase()+'</button>').join('');document.querySelectorAll('#devices button').forEach(x=>x.onclick=()=>{device=x.dataset.device;render()});seek.oninput=()=>{stop();pos=Number(seek.value);sync()};document.getElementById('toggle').onclick=()=>play(which);tables();render()</script></html>'''
html = html.replace('__DATA__', json.dumps(payload, ensure_ascii=False, separators=(',', ':')))
(PUBLIC / 'speaker-guitar-confidence-eq.html').write_text(html + '\n')
