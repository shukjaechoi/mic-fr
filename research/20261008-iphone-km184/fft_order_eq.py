"""Direct relative-magnitude FFT FIR order and gain-cap probe.
Same8 design/5 previously inspected evaluation hits; no evaluation tuning.
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
variants=('latest','percussion','fir1025','fir4097','fir8193','fir8193strong');models={};summary={};all_events=[];dense=np.geomspace(500,14000,800)
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

def process(d,a,b,key):
 m=models[d][key]
 if 'sos' in m:return read(d,a,b,sos=np.array(m['sos']))
 left=a-.5;right=b+.5;x=read(d,left,right,sos=np.array(m['base_sos']));y=signal.fftconvolve(x,np.array(m['fir']),mode='same');st=mapped(d,a)-mapped(d,left);end=mapped(d,b)-mapped(d,left)
 return y[st:end]
def actual_response(d,key,hz):
 m=models[d][key]
 if 'sos' in m:return response(np.array(m['sos']),hz)
 return response(np.array(m['base_sos']),hz)+20*np.log10(np.maximum(abs(signal.freqz(np.array(m['fir']),worN=hz,fs=FS)[1]),1e-12))
# One direct magnitude target shared across tap counts; ±12 probe isolates
# the gain constraint separately, rather than attributing it to FIR order.
ns=1000*2**(np.arange(-18,90)/24);ns=ns[(ns>=600)&(ns<=12000)]
for d in ('iphone','s21'):
 target,rel,support,mad=obs(d,ns,1/24,latest[d]);valid=(np.array(support)>=4)&(rel>=.1)
 curve=target*(.5+.5*np.clip(rel/.5,0,1));curve[~valid]=0;curve=ndimage.gaussian_filter1d(curve,.7)
 # Missing unresolved low bands pull toward zero; never turn FFT padding into evidence.
 freq=np.linspace(0,FS/2,32769);needed=np.interp(freq,ns,curve,left=0,right=0);taper=np.minimum(np.clip((freq-500)/200,0,1),np.clip((14000-freq)/2000,0,1));needed*=taper
 models[d]={'latest':{'sos':latest[d].tolist()},'percussion':old_perc['models'][d]}
 for key,taps,limit in [('fir1025',1025,6),('fir4097',4097,6),('fir8193',8193,6),('fir8193strong',8193,12)]:
  gain=10**(np.clip(needed,-limit,limit)/20);h=signal.firwin2(taps,freq,gain,fs=FS,nfreqs=32769,window=('kaiser',6));assert np.allclose(h,h[::-1],atol=1e-12)
  name=f'fft-filter-{d}-{key}.wav';sf.write(OUT/name,h,FS,subtype='FLOAT')
  models[d][key]={'fir':h.tolist(),'base_sos':latest[d].tolist(),'taps':taps,'delay_ms':(taps-1)/2/FS*1000,'target_limit_db':limit,'filter_wav':name,'sha256':hashlib.sha256(h.tobytes()).hexdigest(),'target_centers_hz':ns.tolist(),'raw_target_db':target.tolist(),'target_reliability':rel.tolist(),'support_hits':support,'mad_db':mad}
 for key,m in models[d].items():
  if 'sos' in m:
   sos=np.array(m['sos']);assert all(max(abs(np.roots(z[3:])))<1 for z in sos);m['sha256']=hashlib.sha256(sos.tobytes()).hexdigest()
 summary[d]={};gain=old_perc['models'][d]['fixed_train_gain_db']
 for e in events:
  rec=e['devices'][d];valid=rec['weight']>=.15;scores={};signed={}
  for key in variants:
   pp=spectrum(process(d,e['start_s'],e['end_s'],key));err=10*np.log10(bands(rec['ref'])/bands(pp))-gain;signed[key]=err.tolist();scores[key]={}
   for region,rr in [('600_12000',np.ones(len(centers),bool)),('800_2000',(centers>=780)&(centers<=2050)),('8000_12000',centers>=8000)]:
    take=rr&valid;scores[key][region]=float(np.average(abs(err[take]),weights=rec['weight'][take])) if np.any(take) else None
  all_events.append({'device':d,'id':e['id'],'start_s':e['start_s'],'end_s':e['end_s'],'split':e['split'],'weight':rec['weight'].tolist(),'signed':signed,'scores':scores})
 for split in ('design','held_hit'):
  rows=[e for e in all_events if e['device']==d and e['split']==split];summary[d][split]={}
  for key in variants:
   summary[d][split][key]={}
   for region in ('600_12000','800_2000','8000_12000'):
    pairs=[(e['scores']['latest'][region],e['scores'][key][region]) for e in rows if e['scores'][key][region] is not None and e['scores']['latest'][region] is not None];summary[d][split][key][region]={'mae_db':float(np.mean([v[1] for v in pairs])) if pairs else None,'hits':len(pairs),'worse_hits_gt_0_5db':sum(v[1]-v[0]>.5 for v in pairs)}
report={'scope':'FFT FIR order/gain-limit percussion EQ development; same first8 design/final5 evaluation hits, already inspected in prior experiments. Not independent validation.','rules':{'candidates':'Same direct 1/24 octave observable residual magnitude curve; 1025/4097/8193 linear-phase FIR at ±6dB, plus8193 ±12dB probe. Same8 design hits, no evaluation retuning.','limits':'Target clip ±6 or ±12dB, same curve otherwise. Actual realized response may deviate due window/truncation; measured explicitly. Below500/above14k target unity.','confidence':'Raw-device and KM184 SNR, relevance, physical duration resolution, >=4/8 observed hits, sign consistency and MAD shrinkage. Same original 1/12 masks for every score.','gain':'Same previous fixed train-derived 200–600Hz gain in all event evaluations.','scope':'Magnitude-only FIR, Kaiser6 window; group delay10.667/42.667/85.333ms compensated offline. Zero-padding not observation resolution. Linear-phase pre-ringing possible.'},'models':models,'summary':summary,'events':all_events,'gain_db':{d:old_perc['models'][d]['fixed_train_gain_db'] for d in models}}
(OUT/'fft-order-eq.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
# Continuous-context WAVs, rendered with identical clip level and colour scales.
r=json.loads(json.dumps(browser));r['mixed_experiment']=report
freqgrid=np.geomspace(12000,80,320)
def spec(x):
 ff,t,z=signal.stft(x,FS,nperseg=4096,noverlap=3584,boundary=None);fi,ti=np.meshgrid(freqgrid/(ff[1]-ff[0]),(np.linspace(0,5,470)-t[0])/(t[1]-t[0]),indexing='ij');return ndimage.map_coordinates(20*np.log10(np.maximum(abs(z),1e-8)),[fi,ti],order=1,mode='nearest')
for d in models:
 r['models'][d]['selected']='fir8193strong';r['models'][d]['application_scope']['percussion_option']='fir8193strong';r['response_db'][d]={}
 for key in variants:
  ui='oct_12' if key=='latest' else key;r['response_db'][d][ui]=actual_response(d,key,np.array(r['response_hz'])).tolist()
for clip in r['clips']:
 d=clip['device'];a=clip['start_s'];waves={'km184':read('km184',a,a+5),'raw':read(d,a,a+5)}
 for key in variants:waves['oct_12' if key=='latest' else key]=process(d,a,a+5,key)
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
  stem=f'fftorder-{clip["group"]}-{a}-{d}-{k}';sf.write(OUT/(stem+'.wav'),v,FS,subtype='PCM_16');Image.fromarray(np.uint8(colormaps['magma'](np.clip((spectra[k]-top+72)/72,0,1))[:,:,:3]*255)).save(OUT/(stem+'.png'),optimize=True);media={'audio':stem+'.wav','spectrogram':stem+'.png'}
  if k!='km184':
   z=ndimage.gaussian_filter(spectra[k],(1.2,1));rgb=colormaps['RdBu_r'](np.clip((z-ref+12)/24,0,1))[:,:,:3];rgb[(ref<top-58)|(z<top-58)]=[.88,.9,.91];Image.fromarray(np.uint8(rgb*255)).save(OUT/(stem+'-error.png'),optimize=True);media['residual']=stem+'-error.png'
  clip['media'][k]=media
 print('rendered',d,clip['group'],flush=True)
(OUT/'fft-order-browser.json').write_text(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
html=(OUT/'percussion-response-eq.html').read_text();a=html.index('<script>');b=html.index('</script>',a);js=html[a+8:b];js=js[js.index(';const names='):]
js=js.replace('min=fullCurve?-14:-6,max=fullCurve?8:6','min=fullCurve?-24:-14,max=14').replace('fullCurve?[-10,-5,0,5]:[-4,-2,0,2,4]','fullCurve?[-20,-10,0,10]:[-12,-6,0,6,12]')
js=js.replace("variant='percussion'","variant='fir8193strong'").replace("const labels={", "const labels={fir1025:'FIR 1,025탭 · ±6dB',fir4097:'FIR 4,097탭 · ±6dB',fir8193:'FIR 8,193탭 · ±6dB',fir8193strong:'FIR 8,193탭 · ±12dB',").replace("const colors={","const colors={fir1025:'#cf8132',fir4097:'#4b7dc1',fir8193:'#038e83',fir8193strong:'#bd3e5d',")
js=js.replace("let h='';for(const [key,title]", "let h='<tr><td>600Hz–12kHz 예시 오차</td><td>'+c.fullband_scores.oct_12.toFixed(2)+'dB</td><td>'+c.fullband_scores[variant].toFixed(2)+'dB</td><td>'+(c.fullband_scores[variant]-c.fullband_scores.oct_12).toFixed(2)+'dB</td></tr>';for(const [key,title]")
x=js.index("document.getElementById('status').textContent=");y=js.index(";document.getElementById('candidates')",x);js=js[:x]+"document.getElementById('status').textContent=group==='percussion'?'같은 목표 응답·±6dB에서 탭 수를 비교하고, ±12dB는 제한 완화만 별도로 시험합니다. 청취 지연은 맞췄습니다. 이 두 타격은 설계에서 제외했습니다.':'퍼커션 기반 FFT FIR을 이 주법에 그대로 적용합니다. 세밀한 보정과 강한 후보의 부작용도 함께 비교하세요.'"+js[y:]
html=html[:a]+'<script>const data='+json.dumps(r,ensure_ascii=False,separators=(',',':'))+js+'</script>'+html[b+9:]
html=html.replace('<title>퍼커션 지속 구간 기반 상대 응답 EQ</title>','<title>FFT FIR 탭 수와 보정 제한 비교</title>').replace('<h1>퍼커션 지속 구간 기반 보정 vs 마지막 1/12 필터</h1>','<h1>FFT FIR 1,025 · 4,097 · 8,193탭 비교</h1>');x=html.index('<p class="lead">');y=html.index('</p>',x)+4
html=html[:x]+'<p class="lead">피킹 EQ의 근사 대신, 퍼커션 상대 크기 응답을 <b>FFT 컨볼루션 FIR</b>로 직접 구현했습니다. <b>같은 목표·±6dB에서 1,025/4,097/8,193탭</b>을 비교하고, 8,193탭 ±12dB는 보정 제한 완화의 별도 진단입니다. 기존 저역·유음용 EQ 위에 적용하며, 실제 녹음의 앞뒤 문맥을 포함해 처리하고 FIR 지연은 청취에서 맞췄습니다. 긴 FIR도 새로운 관측 정보를 만들거나 공간·위상 차이를 복원하지는 않습니다. <b>이번 수치에서 탭 수 증가의 이점은 매우 작았고, S21에서는 ±12dB 제한 완화가 더 큰 영향을 보였습니다.</b></p>'+html[y:]
html=html.replace('<nav>','<nav><a href="percussion-response-eq.html">이전 퍼커션 보완</a>',1)
x=html.index('<h2>퍼커션에서 추정한 상대 FR와 실제 보완</h2>');y=html.index('<script>',x);labels={'latest':'마지막 유음용 1/12','percussion':'이전 퍼커션 1/12','fir1025':'FIR 1,025 · ±6','fir4097':'FIR 4,097 · ±6','fir8193':'FIR 8,193 · ±6','fir8193strong':'FIR 8,193 · ±12'};rows=''
for d in models:
 for key in variants:
  vv=summary[d]['held_hit'][key];rows+=f'<tr><td>{d.upper()}</td><td>{labels[key]}</td>'+''.join(f'<td>{vv[region]["mae_db"]:.2f}</td>' for region in ('600_12000','800_2000','8000_12000'))+f'<td>{vv["600_12000"]["worse_hits_gt_0_5db"]}/5</td></tr>'
cross=''.join(f'<tr><td>{c["device"].upper()} · {c["label"]}</td>'+''.join(f'<td>{c["fullband_scores"][k]:.2f}</td>' for k in ('oct_12','percussion','fir1025','fir4097','fir8193','fir8193strong'))+'</tr>' for c in r['clips'])
content='<h2>同じ評価5타격에서 비교</h2><p>600Hz–12kHz와 각 부분 대역의 신뢰도 가중 평균 절대 오차(dB), 작을수록 가깝습니다. 설계 8타격에서 정한 200–600Hz 고정 게인을 모든 후보에 유지합니다. 악화는 마지막 유음용 1/12 대비 타격 평균 오차가 0.5dB 넘게 증가한 경우입니다. 자료와 기존 필터 모두 탐색 노출이 있어 독립 검증이 아닙니다.</p><div class="table-wrap"><table><tr><th>기기</th><th>후보</th><th>600Hz–12kHz</th><th>800Hz–2kHz</th><th>8–12kHz</th><th>악화 타격</th></tr>'+rows+'</table></div><details><summary>다른 주법 예시의 적용 범위</summary><p>각 5초 예시는 기존 원본 기반 레벨 기준과 원본 기반 관측 마스크를 공유합니다. 위 타격 평가와 다른 기준이므로 절댓값을 합치지 않습니다.</p><div class="table-wrap"><table><tr><th>기기·주법</th><th>마지막 유음용</th><th>이전 퍼커션</th><th>1,025 · ±6</th><th>4,097 · ±6</th><th>8,193 · ±6</th><th>8,193 · ±12</th></tr>'+cross+'</table></div></details><details><summary>설계 방식과 필터 계수</summary><p>직접 목표는 설계 8타격의 1/24옥타브 중앙 크기 잔차입니다. 타격 길이로 분해할 수 없거나 관측이 약한 성분은 무보정 쪽으로 줄입니다. Kaiser(6) 창의 선형위상 FIR로 구현하고 실제 응답을 비교합니다. 양쪽 신호의 SNR·에너지 비중·타격 지속시간·최소4/8타격 관측·부호 일치·MAD를 사용하며, 약한 관측은 낮게 가중합니다. 탭 수 비교는 동일 ±6dB 목표이며, ±12dB는 강도 제한 진단입니다. 실제 구현 응답은 아래 그래프에서 확인합니다. 선형위상 지연은 10.667/42.667/85.333ms이며 재생 비교에서 보상했습니다. 타격 이전에 선형위상 프리링잉이 생길 수 있습니다. 600Hz 이하 기존 저역을 대체하지 않습니다. 실제 필터는 연속 문맥으로 적용했고, 모든 버전의 청취 게인·안전 배율·스펙트로그램 색 범위가 같습니다. 이 실험은 크기 응답 비교이며, 원래의 공간·시간·위상 차이를 추정해 복원한 것은 아닙니다. 필터 IR WAV는 FLOAT이며 계수를 정규화하지 않았습니다.</p><a href="fft-order-eq.json">FIR 계수·필터 WAV 경로·신뢰도·타격별 잔차 JSON</a></details>'
downloads=''.join('<p>'+d.upper()+' · '+ ' / '.join('<a href="'+m['filter_wav']+'">'+labels[k]+'</a>' for k,m in models[d].items() if 'filter_wav' in m)+'</p>' for d in models)
html=html[:x]+content+'<details><summary>추가 FIR 임펄스 WAV 받기</summary><p>48kHz FLOAT 추가 필터 계수이며, 마지막 유음용 EQ 뒤에 적용하는 보완단입니다. 전체 처리 설정과 기본 SOS는 JSON에 있습니다.</p>'+downloads+'</details>'+html[y:];html=html.replace('同じ評価5타격','같은 평가 5타격');(OUT/'fft-order-eq.html').write_text(html+'\n');print(json.dumps(summary,indent=2),flush=True)
