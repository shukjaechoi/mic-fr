"""Fixed coarse-plus-residual EQ comparison using the same percussion split.
Three predeclared constructions: broad 1/3 alone, broad+1/6, broad+1/12.
No held-hit retuning; held hits have already been inspected, so exploratory.
"""
from pathlib import Path
import json, hashlib, re
import numpy as np
from scipy import signal, optimize, ndimage
from PIL import Image
from matplotlib import colormaps
import soundfile as sf
# Reuse the existing input/gate definitions only; do not execute its fitting,
# rendering, result saving, or publishing steps.
source=Path(__file__).with_name('percussion_response_eq.py').read_text()
exec(compile(source[:source.index('models={};results={};')],str(Path(__file__).with_name('percussion_response_eq.py')),'exec'))
old_perc=json.loads((OUT/'percussion-response-eq.json').read_text());browser=json.loads((OUT/'percussion-response-browser.json').read_text());common_centers=centers.copy();common_masks=list(masks)
variants=('latest','percussion','wide','mix6','mix12');models={};summary={};all_events=[];dense=np.geomspace(500,14000,800)
def bandmeans(p,mm):return np.array([np.mean(p[m]) for m in mm])
def obs(d,ns,width,base_sos):
 mm=[(f>=c/2**(width/2))&(f<c*2**(width/2)) for c in ns];vals=[];ws=[]
 for e in events[:8]:
  rec=e['devices'][d];kp,xp=bandmeans(rec['ref'],mm),bandmeans(rec['raw'],mm);pp=spectrum(read(d,e['start_s'],e['end_s'],sos=base_sos));snr=np.minimum(10*np.log10(kp/bandmeans(floor['km184'],mm)),10*np.log10(xp/bandmeans(floor[d],mm)))
  strongest=lambda p:max(np.sum(p[(f>=lo)&(f<hi)])*(f[1]-f[0]) for lo,hi in [(100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6400),(6400,12000)])
  relevance=np.minimum(10*np.log10(kp*ns*.707/strongest(rec['ref'])),10*np.log10(xp*ns*.707/strongest(rec['raw'])));resolved=ns*(2**(width/2)-2**(-width/2))>=3/e['duration_s'];w=np.clip((snr-10)/15,0,1)*(.2+.8*np.clip((relevance+36)/24,0,1))*resolved
  vals.append(10*np.log10(kp/bandmeans(pp,mm))-old_perc['models'][d]['fixed_train_gain_db']);ws.append(w)
 vals=np.array(vals);ws=np.array(ws);target=[];rel=[];support=[];mad=[]
 for j in range(len(ns)):
  vv=vals[:,j][ws[:,j]>=.15];n=len(vv);med=float(np.median(vv)) if n else 0;spread=float(np.median(abs(vv-med))) if n else 0;sgn=float(np.mean(np.sign(vv)==np.sign(med))) if n else 0
  target.append(med);rel.append(n/8*max(0,2*sgn-1)*float(np.median(ws[:,j]))/(1+spread/4) if n>=4 else 0);support.append(n);mad.append(spread)
 return np.array(target),np.array(rel),support,mad

def fit(d,res,base_sos,limit):
 ns=1000*2**(np.arange(-20,50)/res);ns=ns[(ns>=600)&(ns<=12000)];target,rel,support,mad=obs(d,ns,1/res,base_sos);q=np.sqrt(2**(1/res))/(2**(1/res)-1);unit=np.stack([response(np.array([peak(c,q,1)]),ns) for c in ns],axis=1)
 grid=np.geomspace(500,14000,100);h=np.log2(grid[1]/grid[0]);shape=np.stack([response(np.array([peak(c,q,1)]),grid) for c in ns],axis=1);curve=np.diff(shape,n=2,axis=0)/h**1.5;outside=np.stack([response(np.array([peak(c,q,1)]),[100,200,400,500,15000,20000]) for c in ns],axis=1)
 A=np.vstack([unit*np.sqrt(rel[:,None]),.002*curve,.8*outside,.15*np.eye(len(ns))]);b=np.r_[np.clip(target,-limit,min(limit,4))*np.sqrt(rel),np.zeros(len(A)-len(ns))];g=optimize.lsq_linear(A,b,bounds=(-limit,min(limit,4))).x;sos=np.array([peak(c,q,v) for c,v in zip(ns,g)])
 return sos,{'resolution':res,'centers_hz':ns.tolist(),'q':float(q),'gains_db':g.tolist(),'target_db':target.tolist(),'reliability':rel.tolist(),'support_hits':support,'mad_db':mad}

def cap(stages):
 # All constructions have the same total supplementary ±6dB response cap.
 stages=json.loads(json.dumps(stages));sos=np.vstack([np.array([peak(c,st['q'],g) for c,g in zip(st['centers_hz'],st['gains_db'])]) for st in stages])
 for _ in range(4):
  mx=max(abs(response(sos,dense)))
  if mx<=6.001:break
  for st in stages:st['gains_db']=[g*6/mx for g in st['gains_db']]
  sos=np.vstack([np.array([peak(c,st['q'],g) for c,g in zip(st['centers_hz'],st['gains_db'])]) for st in stages])
 return sos,stages

for d in ('iphone','s21'):
 wide,design=fit(d,3,latest[d],6);wide,stages=cap([design]);models[d]={'latest':{'sos':latest[d].tolist()},'percussion':old_perc['models'][d]}
 models[d]['wide']={'sos':np.vstack([latest[d],wide]).tolist(),'stages':stages}
 for res,key in [(6,'mix6'),(12,'mix12')]:
  # Estimate only what remains after the common broad correction. It is not
  # an average of independently fitted filters that might double-correct.
  fine,fd=fit(d,res,np.vstack([latest[d],wide]),3);extra,ss=cap(stages+[fd]);models[d][key]={'sos':np.vstack([latest[d],extra]).tolist(),'stages':ss}
 for key,m in models[d].items():
  sos=np.array(m['sos']);assert all(max(abs(np.roots(z[3:])))<1 for z in sos);m['sha256']=hashlib.sha256(sos.tobytes()).hexdigest()
 summary[d]={};gain=old_perc['models'][d]['fixed_train_gain_db']
 for e in events:
  rec=e['devices'][d];valid=rec['weight']>=.15;scores={};signed={}
  for key in variants:
   pp=spectrum(read(d,e['start_s'],e['end_s'],sos=np.array(models[d][key]['sos'])));err=10*np.log10(bands(rec['ref'])/bands(pp))-gain;signed[key]=err.tolist();scores[key]={}
   for region,rr in [('600_12000',np.ones(len(centers),bool)),('800_2000',(centers>=780)&(centers<=2050)),('8000_12000',centers>=8000)]:
    take=rr&valid;scores[key][region]=float(np.average(abs(err[take]),weights=rec['weight'][take])) if np.any(take) else None
  all_events.append({'device':d,'id':e['id'],'start_s':e['start_s'],'end_s':e['end_s'],'split':e['split'],'weight':rec['weight'].tolist(),'signed':signed,'scores':scores})
 for split in ('design','held_hit'):
  rows=[e for e in all_events if e['device']==d and e['split']==split];summary[d][split]={}
  for key in variants:
   summary[d][split][key]={}
   for region in ('600_12000','800_2000','8000_12000'):
    pairs=[(e['scores']['latest'][region],e['scores'][key][region]) for e in rows if e['scores'][key][region] is not None and e['scores']['latest'][region] is not None];summary[d][split][key][region]={'mae_db':float(np.mean([v[1] for v in pairs])) if pairs else None,'hits':len(pairs),'worse_hits_gt_0_5db':sum(v[1]-v[0]>.5 for v in pairs)}
report={'scope':'Mixed-resolution percussion EQ development; same first8 design/final5 evaluation hits, already inspected in prior experiments. Not independent validation.','rules':{'candidates':'Frozen latest low/mid EQ + broad1/3 only; broad1/3 then remaining1/6 residual; broad1/3 then remaining1/12 residual. No retuning on evaluation hits.','limits':'Same extra total ±6dB response cap; fine residual sections ±3dB; under600Hz approximately preserved.','confidence':'Raw-device and KM184 SNR, relevance, physical duration resolution, >=4/8 observed hits, sign consistency and MAD shrinkage. Same original 1/12 masks for every score.','gain':'Same previous fixed train-derived 200–600Hz gain in all event evaluations.','scope':'Magnitude-only EQ; mixing widths is not time-envelope or phase restoration.'},'models':models,'summary':summary,'events':all_events,'gain_db':{d:old_perc['models'][d]['fixed_train_gain_db'] for d in models}}
(OUT/'mixed-resolution-eq.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
# Continuous-context WAVs, rendered with identical clip level and colour scales.
r=json.loads(json.dumps(browser));r['mixed_experiment']=report
freqgrid=np.geomspace(12000,80,320)
def spec(x):
 ff,t,z=signal.stft(x,FS,nperseg=4096,noverlap=3584,boundary=None);fi,ti=np.meshgrid(freqgrid/(ff[1]-ff[0]),(np.linspace(0,5,470)-t[0])/(t[1]-t[0]),indexing='ij');return ndimage.map_coordinates(20*np.log10(np.maximum(abs(z),1e-8)),[fi,ti],order=1,mode='nearest')
for d in models:
 r['models'][d]['selected']='mix12';r['models'][d]['application_scope']['percussion_option']='mix12';r['response_db'][d]={}
 for key in variants:
  ui='oct_12' if key=='latest' else key;r['response_db'][d][ui]=response(np.array(models[d][key]['sos']),np.array(r['response_hz'])).tolist()
for clip in r['clips']:
 d=clip['device'];a=clip['start_s'];waves={'km184':read('km184',a,a+5),'raw':read(d,a,a+5)}
 for key in variants:waves['oct_12' if key=='latest' else key]=read(d,a,a+5,sos=np.array(models[d][key]['sos']))
 waves={k:signal.resample(v,240000) if len(v)!=240000 else v for k,v in waves.items()};rp=spectrum(waves['km184']);xp=spectrum(waves['raw']);ww,snr,rs=confidence(rp,xp,d,5);valid=ww>=.15;clip['fullband_scores']={};clip['results']={}
 mf=np.fft.rfftfreq(8192,1/FS);common=np.array(r['common_evaluation_centers_hz']);mm=[(mf>=c/2**(1/24))&(mf<c*2**(1/24)) for c in common];welch=lambda v:signal.welch(v,FS,nperseg=8192,noverlap=4096)[1];baseps=welch(waves['km184']);hm=(mf>=8000)&(mf<12000)
 for k,v in list(waves.items()):
  if k not in ('km184','raw'):
   err=10*np.log10(bands(rp)/bands(spectrum(v)))-clip['anchor_db'];clip['fullband_scores'][k]=float(np.average(abs(err[valid]),weights=ww[valid]));pp=welch(v);ce=np.array([10*np.log10(np.mean(baseps[m])/np.mean(pp[m]))-clip['anchor_db'] for m in mm]);weight=np.array(clip['eval_confidence']);sc={}
   for region,take in [('mid_all',np.ones(len(common),bool)),('800_1000',common<=1030),('1500_2000',common>=1450)]:
    take=take&(weight>=.15);sc[region]=float(np.average(abs(ce[take]),weights=weight[take])) if np.any(take) else None
   clip['results'][k]={'scores':sc,'signed_km_minus_device_db':ce.tolist(),'8000_12000_signed_db':float(10*np.log10(np.mean(baseps[hm])/np.mean(pp[hm]))-clip['anchor_db'])}
  if k!='km184':waves[k]=v*10**(clip['anchor_db']/20)
 scale=min(1,.8/max(np.max(abs(v)) for v in waves.values()));waves={k:v*scale for k,v in waves.items()};clip['common_safety_scale']=float(scale);spectra={k:spec(v) for k,v in waves.items()};top=max(z.max() for z in spectra.values());ref=ndimage.gaussian_filter(spectra['km184'],(1.2,1));clip['media']={}
 for k,v in waves.items():
  stem=f'mixeq-{clip["group"]}-{a}-{d}-{k}';sf.write(OUT/(stem+'.wav'),v,FS,subtype='PCM_16');Image.fromarray(np.uint8(colormaps['magma'](np.clip((spectra[k]-top+72)/72,0,1))[:,:,:3]*255)).save(OUT/(stem+'.png'),optimize=True);media={'audio':stem+'.wav','spectrogram':stem+'.png'}
  if k!='km184':
   z=ndimage.gaussian_filter(spectra[k],(1.2,1));rgb=colormaps['RdBu_r'](np.clip((z-ref+12)/24,0,1))[:,:,:3];rgb[(ref<top-58)|(z<top-58)]=[.88,.9,.91];Image.fromarray(np.uint8(rgb*255)).save(OUT/(stem+'-error.png'),optimize=True);media['residual']=stem+'-error.png'
  clip['media'][k]=media
 print('rendered',d,clip['group'],flush=True)
(OUT/'mixed-resolution-browser.json').write_text(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
html=(OUT/'percussion-response-eq.html').read_text();a=html.index('<script>');b=html.index('</script>',a);js=html[a+8:b];js=js[js.index(';const names='):]
js=js.replace("variant='percussion'","variant='mix12'").replace("const labels={", "const labels={wide:'넓은 1/3 보완',mix6:'1/3 + 1/6 잔차',mix12:'1/3 + 1/12 잔차',").replace("const colors={","const colors={wide:'#cf8132',mix6:'#4b7dc1',mix12:'#038e83',")
js=js.replace("let h='';for(const [key,title]", "let h='<tr><td>600Hz–12kHz 예시 오차</td><td>'+c.fullband_scores.oct_12.toFixed(2)+'dB</td><td>'+c.fullband_scores[variant].toFixed(2)+'dB</td><td>'+(c.fullband_scores[variant]-c.fullband_scores.oct_12).toFixed(2)+'dB</td></tr>';for(const [key,title]")
x=js.index("document.getElementById('status').textContent=");y=js.index(";document.getElementById('candidates')",x);js=js[:x]+"document.getElementById('status').textContent=group==='percussion'?'같은 8타격으로 넓은 보정과 남은 잔차 보정을 단계적으로 설계했습니다. 이 예시의 두 타격은 새 설계에서 제외했습니다.':'퍼커션 기반 혼합 보정을 이 주법에 그대로 적용합니다. 개선·악화 여부를 함께 비교하세요.'"+js[y:]
html=html[:a]+'<script>const data='+json.dumps(r,ensure_ascii=False,separators=(',',':'))+js+'</script>'+html[b+9:]
html=html.replace('<title>퍼커션 지속 구간 기반 상대 응답 EQ</title>','<title>넓은 EQ + 세밀한 잔차 EQ 비교</title>').replace('<h1>퍼커션 지속 구간 기반 보정 vs 마지막 1/12 필터</h1>','<h1>넓은 1/3 EQ + 1/6·1/12 잔차 보정</h1>');x=html.index('<p class="lead">');y=html.index('</p>',x)+4
html=html[:x]+'<p class="lead">단순히 여러 EQ를 평균내지 않고 <b>넓은 1/3 보정 → 그 뒤에 반복적으로 남는 잔차만 1/6 또는 1/12로 보정</b>합니다. 마지막 유음용 1/12 후보와 이전 퍼커션 1/12 후보도 함께 비교합니다. 같은 설계 8타격·평가 5타격·게인·마스크를 유지했습니다. 평가 타격은 이미 살펴본 자료이므로 탐색 결과이며, 혼합 해상도가 모든 주법을 개선하거나 시간 모양을 복원한다는 뜻은 아닙니다. <b>이번 평가에서 혼합 후보의 전체 오차는 이전 퍼커션 후보와 거의 같거나 조금 나빴습니다. iPhone의 8–12kHz만 소폭 유리해, 혼합을 새로운 권장 필터로 채택하지는 않습니다.</b></p>'+html[y:]
html=html.replace('<nav>','<nav><a href="percussion-response-eq.html">이전 퍼커션 보완</a>',1)
x=html.index('<h2>퍼커션에서 추정한 상대 FR와 실제 보완</h2>');y=html.index('<script>',x);labels={'latest':'마지막 유음용 1/12','percussion':'이전 퍼커션 1/12','wide':'넓은 1/3','mix6':'1/3 + 1/6','mix12':'1/3 + 1/12'};rows=''
for d in models:
 for key in variants:
  vv=summary[d]['held_hit'][key];rows+=f'<tr><td>{d.upper()}</td><td>{labels[key]}</td>'+''.join(f'<td>{vv[region]["mae_db"]:.2f}</td>' for region in ('600_12000','800_2000','8000_12000'))+f'<td>{vv["600_12000"]["worse_hits_gt_0_5db"]}/5</td></tr>'
cross=''.join(f'<tr><td>{c["device"].upper()} · {c["label"]}</td>'+''.join(f'<td>{c["fullband_scores"][k]:.2f}</td>' for k in ('oct_12','percussion','wide','mix6','mix12'))+'</tr>' for c in r['clips'])
content='<h2>同じ評価5타격에서 비교</h2><p>600Hz–12kHz와 각 부분 대역의 신뢰도 가중 평균 절대 오차(dB), 작을수록 가깝습니다. 설계 8타격에서 정한 200–600Hz 고정 게인을 모든 후보에 유지합니다. 악화는 마지막 유음용 1/12 대비 타격 평균 오차가 0.5dB 넘게 증가한 경우입니다. 자료와 기존 필터 모두 탐색 노출이 있어 독립 검증이 아닙니다.</p><div class="table-wrap"><table><tr><th>기기</th><th>후보</th><th>600Hz–12kHz</th><th>800Hz–2kHz</th><th>8–12kHz</th><th>악화 타격</th></tr>'+rows+'</table></div><details><summary>다른 주법 예시의 적용 범위</summary><p>각 5초 예시는 기존 원본 기반 레벨 기준과 원본 기반 관측 마스크를 공유합니다. 위 타격 평가와 다른 기준이므로 절댓값을 합치지 않습니다.</p><div class="table-wrap"><table><tr><th>기기·주법</th><th>마지막 유음용</th><th>이전 퍼커션</th><th>넓은 1/3</th><th>1/3+1/6</th><th>1/3+1/12</th></tr>'+cross+'</table></div></details><details><summary>설계 방식과 필터 계수</summary><p>넓은 단계는 실제 1/3옥타브 폭의 피킹 EQ입니다. 세밀한 단계는 넓은 보정의 실제 출력에서 잔차를 다시 측정하므로 이중 보정을 피합니다. 양쪽 신호의 SNR·에너지 비중·타격 지속시간·최소4/8타격 관측·부호 일치·MAD를 사용하며, 약한 관측은 낮게 가중합니다. 세밀한 잔차는 각 단 ±3dB, 전체 추가 응답은 ±6dB로 제한합니다. 600Hz 이하 기존 저역을 대체하지 않습니다. 실제 필터는 연속 문맥으로 적용했고, 모든 버전의 청취 게인·안전 배율·스펙트로그램 색 범위가 같습니다. 이 실험은 크기 응답 비교이며, 시간 엔벌로프와 위상 차이는 그대로 남을 수 있습니다.</p><a href="mixed-resolution-eq.json">단계별 계수·신뢰도·타격별 잔차 JSON</a></details>'
html=html[:x]+content+html[y:];html=html.replace('同じ評価5타격','같은 평가 5타격');(OUT/'mixed-resolution-eq.html').write_text(html+'\n');print(json.dumps(summary,indent=2),flush=True)
