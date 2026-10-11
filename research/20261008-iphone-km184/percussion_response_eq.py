"""Percussion-duration relative magnitude EQ; chronological held-hit comparison.
No IR inversion or phase recovery: estimate a robust magnitude ratio over shared
actual event gates and fit one bounded causal SOS supplement to the frozen latest EQ.
"""
from pathlib import Path
import json, hashlib, re
import numpy as np
import soundfile as sf
from scipy import signal, ndimage, optimize
from PIL import Image
from matplotlib import colormaps
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/experiments/20261008-iphone-km184-public';LOCAL=ROOT/'docs/experiments/20261008-iphone-km184';FS=48000;N=65536
prior=json.loads((OUT/'local-resolution-eq-browser.json').read_text());temporal=json.loads((OUT/'temporal-spread-qc.json').read_text());bg=json.loads((LOCAL/'guitar-background-v2.json').read_text());old=json.loads((ROOT/'docs/experiments/20261002/data/analysis.json').read_text())
maps={m['id']:m['guitarSync'] for m in old['microphones']};maps['iphone']=json.loads((LOCAL/'analysis.json').read_text())['sets']['guitar']['mapping'];audio={d:np.load(f'/private/tmp/mic-audit-20261008/{d}-guitar.npy',mmap_mode='r') for d in ('km184','iphone','s21')}
def mapped(d,t):
 m=maps[d];return round(t*FS*m['slope']+m.get('interceptSamples',m.get('intercept_samples',0)))
def read(d,a,b,context=.5,sos=None):
 st=mapped(d,a);end=mapped(d,b);left=max(0,st-round(context*FS));x=np.asarray(audio[d][left:end],float)
 return (signal.sosfilt(sos,x) if sos is not None else x)[st-left:]
def response(sos,hz):return 20*np.log10(np.maximum(abs(signal.sosfreqz(sos,worN=hz,fs=FS)[1]),1e-12))
def peak(fc,q,g):
 a=10**(g/40);w=2*np.pi*fc/FS;al=np.sin(w)/(2*q);c=np.cos(w);den=np.array([1+al/a,-2*c,1-al/a]);return np.r_[np.array([1+al*a,-2*c,1-al*a])/den[0],den/den[0]]
def spectrum(x):
 if len(x)>N:return signal.welch(x,FS,nperseg=N,noverlap=N//2,detrend=False)[1]
 # A flat event gate preserves the strike; only the first/last 5ms are tapered.
 w=np.ones(len(x));n=min(240,len(x)//10);ramp=np.sin(np.linspace(0,np.pi/2,n))**2;w[:n]=ramp;w[-n:]=ramp[::-1]
 return signal.periodogram(x,FS,window=w,nfft=N,detrend=False)[1]
f=np.fft.rfftfreq(N,1/FS);ff=np.fft.rfftfreq(8192,1/FS)
floor={d:np.interp(f,ff,bg['median_psd'][d]) for d in ('km184','iphone')};floor['s21']=np.interp(f,ff,np.median([signal.welch(read('s21',q['start_s'],q['start_s']+2),FS,nperseg=8192)[1] for q in bg['selected']],axis=0))
centers=1000*2**(np.arange(-9,44)/12);centers=centers[(centers>=600)&(centers<=12000)];masks=[(f>=c/2**(1/24))&(f<c*2**(1/24)) for c in centers];anchor=(f>=200)&(f<600)
def bands(p):return np.array([np.mean(p[m]) for m in masks])
def confidence(k,x,d,duration):
 kp,xp=bands(k),bands(x);snr=np.minimum(10*np.log10(np.maximum(kp/bands(floor['km184']),1e-30)),10*np.log10(np.maximum(xp/bands(floor[d]),1e-30)))
 strongest=lambda p:max(np.sum(p[(f>=lo)&(f<hi)])*(f[1]-f[0]) for lo,hi in [(100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6400),(6400,12000)])
 relevance=np.minimum(10*np.log10(np.maximum(kp*centers*.707/strongest(k),1e-30)),10*np.log10(np.maximum(xp*centers*.707/strongest(x),1e-30)))
 width=centers*(2**(1/24)-2**(-1/24));resolved=width>=3/duration
 return np.clip((snr-10)/15,0,1)*(.2+.8*np.clip((relevance+36)/24,0,1))*resolved,snr,resolved
latest={d:np.array(prior['models'][d]['oct_12']['sos']) for d in ('iphone','s21')};current={'iphone':np.array(json.loads((LOCAL/'speaker-guitar-confidence-eq.json').read_text())['models']['iphone']['sos']),'s21':np.array(json.loads((LOCAL/'brightness-extension.json').read_text())['models']['s21']['sos'])}
hp=signal.butter(3,[700,12000],btype='bandpass',fs=FS,output='sos');events=[]
# Common gate = KM184 onset through the latest of the three observed tails.
# End requires 40ms below both peak-relative (-35dB) and local background (+6dB).
for i,item in enumerate(temporal['events']['km184']):
 hit=item['km184_time_s'];a=hit-.12;b=hit+.85;env={};times=np.arange(round((b-a)*FS))/FS+a
 for d in audio:
  x=read(d,a,b);z=signal.sosfiltfilt(hp,x);e=ndimage.uniform_filter1d(z*z,240);env[d]=np.interp(times,np.arange(len(e))/FS+a,e)
 k=env['km184'];search=(times>=hit-.06)&(times<=hit+.06);pi=np.flatnonzero(search)[np.argmax(k[search])];threshold=k[pi]*10**(-18/10);on=np.flatnonzero((times>=times[pi]-.06)&(times<=times[pi])&(k>=threshold))[0];on=max(0,on-240);ends=[]
 for d,e in env.items():
  peak_energy=np.max(e[search]);quiet=e[:round(.04*FS)];limit=max(peak_energy*10**(-35/10),float(np.median(quiet))*10**(.6));below=e<limit;run=ndimage.uniform_filter1d(below.astype(float),1920,origin=-959);ix=np.flatnonzero((times>hit+.05)&(run>.999));ends.append(float(times[ix[0]]) if len(ix) else b-.04)
 end=min(max(ends)+.01,b-.02);start=float(times[on]);duration=end-start
 assert start>1767.403 and end<1806 and duration>.08
 row={'id':i+1,'peak_s':hit,'start_s':start,'end_s':end,'duration_s':duration,'split':'design' if i<8 else 'held_hit','tail_cap':end>=b-.03,'devices':{}}
 for d in ('iphone','s21'):
  k=spectrum(read('km184',start,end));x=spectrum(read(d,start,end));base=spectrum(read(d,start,end,sos=latest[d]));w,snr,res=confidence(k,x,d,duration)
  row['devices'][d]={'ref':k,'raw':x,'base':base,'weight':w,'snr':snr,'resolved':res}
 events.append(row)
models={};results={};response_hz=np.geomspace(100,16000,600)
for d in ('iphone','s21'):
 train=[e for e in events if e['split']=='design'];gain=float(np.median([10*np.log10(np.mean(e['devices'][d]['ref'][anchor])/np.mean(e['devices'][d]['base'][anchor])) for e in train]))
 errors=np.array([10*np.log10(bands(e['devices'][d]['ref'])/bands(e['devices'][d]['base']))-gain for e in train]);weights=np.array([e['devices'][d]['weight'] for e in train]);targets=[];reliability=[];support=[];mad=[];sign=[]
 for j in range(len(centers)):
  obs=errors[:,j][weights[:,j]>=.15];n=len(obs);med=float(np.median(obs)) if n else 0;spread=float(np.median(abs(obs-med))) if n else 0;consistent=float(np.mean(np.sign(obs)==np.sign(med))) if n else 0
  rel=(n/8)*max(0,2*consistent-1)*float(np.median(weights[:,j]))/(1+spread/4) if n>=4 else 0
  targets.append(np.clip(med,-6,4));reliability.append(rel);support.append(n);mad.append(spread);sign.append(consistent)
 q=np.sqrt(2**(1/12))/(2**(1/12)-1);unit=np.stack([response(np.array([peak(c,q,1)]),centers) for c in centers],axis=1);curv=np.diff(unit,n=2,axis=0);outside=np.stack([response(np.array([peak(c,q,1)]),[100,200,400,500,15000,20000]) for c in centers],axis=1)
 A=np.vstack([unit*np.sqrt(np.array(reliability)[:,None]),.35*curv,.8*outside,.15*np.eye(len(centers))]);target=np.r_[np.array(targets)*np.sqrt(reliability),np.zeros(len(A)-len(centers))];gains=optimize.lsq_linear(A,target,bounds=(-6,4)).x;extra=np.array([peak(c,q,g) for c,g in zip(centers,gains)])
 for _ in range(3):
  mx=max(abs(response(extra,np.geomspace(500,14000,1000))))
  if mx<=6.001:break
  gains*=6/mx;extra=np.array([peak(c,q,g) for c,g in zip(centers,gains)])
 sos=np.vstack([latest[d],extra]);assert all(max(abs(np.roots(z[3:])))<1 for z in sos)
 models[d]={'sos':sos.tolist(),'raw_median_residual_db':[float(np.median(errors[:,j][weights[:,j]>=.15])) if np.any(weights[:,j]>=.15) else None for j in range(len(centers))],'extra_sos':extra.tolist(),'fixed_train_gain_db':gain,'sha256':hashlib.sha256(sos.tobytes()).hexdigest(),'centers_hz':centers.tolist(),'gains_db':gains.tolist(),'target_db':list(map(float,targets)),'reliability':reliability,'support_hits':support,'mad_db':mad,'sign_fraction':sign,'train_hit_ids':[e['id'] for e in train],'held_hit_ids':[e['id'] for e in events if e['split']=='held_hit']}
 for e in events:
  rec=e['devices'][d];scores={};signed={};valid=rec['weight']>=.15
  for name,ss in [('latest',latest[d]),('percussion',sos)]:
   p=spectrum(read(d,e['start_s'],e['end_s'],sos=ss));err=10*np.log10(bands(rec['ref'])/bands(p))-gain;signed[name]=err.tolist();sc={}
   for key,region in [('600_12000',np.ones(len(centers),bool)),('800_2000',(centers>=780)&(centers<=2050)),('8000_12000',centers>=8000)]:
    take=valid&region;sc[key]=float(np.average(abs(err[take]),weights=rec['weight'][take])) if np.any(take) else None
   scores[name]=sc
  rec['scores']=scores;rec['signed']=signed
 results[d]={}
 for split in ('design','held_hit'):
  rows=[e['devices'][d] for e in events if e['split']==split];results[d][split]={}
  for region in ('600_12000','800_2000','8000_12000'):
   pairs=[(x['scores']['latest'][region],x['scores']['percussion'][region]) for x in rows if x['scores']['latest'][region] is not None and x['scores']['percussion'][region] is not None]
   results[d][split][region]={'hits':len(pairs),'latest_mae_db':float(np.mean([p[0] for p in pairs])) if pairs else None,'percussion_mae_db':float(np.mean([p[1] for p in pairs])) if pairs else None,'worse_hits_gt_0_5db':sum(p[1]-p[0]>.5 for p in pairs)}
# Save all event boundaries, masks, and both signed residuals; spectra omitted.
saved=[]
for e in events:
 r={k:v for k,v in e.items() if k!='devices'};r['devices']={}
 for d,rec in e['devices'].items():r['devices'][d]={k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in rec.items() if k not in ('ref','raw','base')}
 saved.append(r)
report={'scope':'October same-session developmental percussion-duration relative magnitude experiment. First 8 hits design, final 5 held from new fit; prior EQ saw this session. No IR or phase inversion.','rules':{'gates':'Common KM184 onset through maximum three-device observed tail, 40ms below peak -35dB/local quiet +6dB; 0.83s cap. 5ms edge taper, no full Hann that suppresses attack.','resolution':'1/12 octave only where physical band width >=3/event duration; zero-padding does not create resolution. Joint raw SNR >=10dB, relevance weights, >=4/8 design-hit support and sign/MAD shrinkage.','gain':'One 200–600Hz gain learned on 8 design hits from latest output, frozen for both filters and all held hits.','filter':'Frozen latest 1/12 filter plus one 600–12k relative-EQ supplement; section -6/+4dB, total supplement absolute cap6dB, causal IIR. Under600Hz held approximately constant.','selection':'One candidate, no held-hit tuning or refitting; developmental comparison, not independent recording validation.','limitation':'Magnitude EQ cannot reconstruct phase, spatial radiation or decay differences. Gate detects acoustic energy, not verified hand-contact duration.'},'models':models,'summary':results,'events':saved,'centers_hz':centers.tolist()}
(OUT/'percussion-response-eq.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
# Real continuous-context clips and a comparison page using the established UI.
r=json.loads(json.dumps(prior));r['percussion_experiment']=report
freqgrid=np.geomspace(12000,80,320)
def spec(x):
 f,t,z=signal.stft(x,FS,nperseg=4096,noverlap=3584,boundary=None);fi,ti=np.meshgrid(freqgrid/(f[1]-f[0]),(np.linspace(0,5,470)-t[0])/(t[1]-t[0]),indexing='ij');return ndimage.map_coordinates(20*np.log10(np.maximum(abs(z),1e-8)),[fi,ti],order=1,mode='nearest')
for d in ('iphone','s21'):
 r['models'][d]['percussion']={'sos':models[d]['sos']};r['models'][d]['selected']='percussion';r['models'][d]['application_scope']['percussion_option']='percussion';r['response_db'][d]={k:v for k,v in r['response_db'][d].items() if k in ('current','high_only','oct_12')};r['response_db'][d]['percussion']=response(np.array(models[d]['sos']),np.array(r['response_hz'])).tolist()
for clip in r['clips']:
 d=clip['device'];a=clip['start_s'];sos=np.array(models[d]['sos']);waves={'km184':read('km184',a,a+5),'raw':read(d,a,a+5),'current':read(d,a,a+5,sos=current[d]),'oct_12':read(d,a,a+5,sos=latest[d]),'percussion':read(d,a,a+5,sos=sos)}
 if d=='s21':
  high=np.array(prior['models'][d]['oct_12']['sos'])[:-len(prior['models'][d]['oct_12']['extra_sos'])];waves['high_only']=read(d,a,a+5,sos=high)
 # Clock mapping changes sample count slightly. Resample all to the KM184
 # common 5-second display timeline (only clock drift, not arbitrary alignment).
 waves={k:signal.resample(v,240000) if len(v)!=240000 else v for k,v in waves.items()};rp=spectrum(waves['km184']);xp=spectrum(waves['raw']);ww,snr,res=confidence(rp,xp,d,5);valid=ww>=.15
 clip['fullband_scores']={};clip['results']={k:v for k,v in clip['results'].items() if k in ('current','high_only','oct_12')}
 mf=np.fft.rfftfreq(8192,1/FS);common=np.array(r['common_evaluation_centers_hz']);mm=[(mf>=c/2**(1/24))&(mf<c*2**(1/24)) for c in common];welch=lambda v:signal.welch(v,FS,nperseg=8192,noverlap=4096)[1];baseps=welch(waves['km184']);hm=(mf>=8000)&(mf<12000)
 for k,v in waves.items():
  if k not in ('km184','raw'):
   err=10*np.log10(bands(rp)/bands(spectrum(v)))-clip['anchor_db'];clip['fullband_scores'][k]=float(np.average(abs(err[valid]),weights=ww[valid]));pp=welch(v);ce=np.array([10*np.log10(np.mean(baseps[m])/np.mean(pp[m]))-clip['anchor_db'] for m in mm]);weight=np.array(clip['eval_confidence']);scores={}
   for key,reg in [('mid_all',np.ones(len(common),bool)),('800_1000',common<=1030),('1500_2000',common>=1450)]:
    take=reg&(weight>=.15);scores[key]=float(np.average(abs(ce[take]),weights=weight[take])) if np.any(take) else None
   clip['results'][k]={'scores':scores,'signed_km_minus_device_db':ce.tolist(),'8000_12000_signed_db':float(10*np.log10(np.mean(baseps[hm])/np.mean(pp[hm]))-clip['anchor_db'])}
  if k!='km184':waves[k]=v*10**(clip['anchor_db']/20)
 scale=min(1,.8/max(np.max(abs(v)) for v in waves.values()));waves={k:v*scale for k,v in waves.items()};clip['common_safety_scale']=float(scale);spectra={k:spec(v) for k,v in waves.items()};top=max(z.max() for z in spectra.values());ref=ndimage.gaussian_filter(spectra['km184'],(1.2,1));clip['media']={}
 for k,v in waves.items():
  stem=f'perceq-{clip["group"]}-{a}-{d}-{k}';sf.write(OUT/(stem+'.wav'),v,FS,subtype='PCM_16');Image.fromarray(np.uint8(colormaps['magma'](np.clip((spectra[k]-top+72)/72,0,1))[:,:,:3]*255)).save(OUT/(stem+'.png'),optimize=True);media={'audio':stem+'.wav','spectrogram':stem+'.png'}
  if k!='km184':
   z=ndimage.gaussian_filter(spectra[k],(1.2,1));rgb=colormaps['RdBu_r'](np.clip((z-ref+12)/24,0,1))[:,:,:3];rgb[(ref<top-58)|(z<top-58)]=[.88,.9,.91];Image.fromarray(np.uint8(rgb*255)).save(OUT/(stem+'-error.png'),optimize=True);media['residual']=stem+'-error.png'
  clip['media'][k]=media
 print('rendered',d,clip['group'],flush=True)
(OUT/'percussion-response-browser.json').write_text(json.dumps(r,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
# Keep shared player and visual inspection behavior, replace experiment-specific text/tables.
html=(OUT/'local-resolution-eq.html').read_text();start=html.index('<script>');end=html.index('</script>',start);js=html[start+8:end];js=js[js.index(';const names='):]
js=js.replace("let group='complex',device='s21',variant='oct_12'","let group='percussion',device='s21',variant='percussion'")
js=js.replace("const labels={current:","const labels={percussion:'퍼커션 기반 후보',current:").replace("const colors={current:","const colors={percussion:'#ad428b',current:").replace("oct_12:'1/12 중역 보완'","oct_12:'마지막 1/12 후보'")
js=js.replace("rawMode?'raw':'current'","rawMode?'raw':'oct_12'").replace("rawMode?'기기 원본':'앞선 청취 후보'","rawMode?'기기 원본':'마지막 1/12 후보'")
js=js.replace('base=data.response_db[device].current','base=data.response_db[device].oct_12')
js=js.replace("let b=c.results.current.scores[key]","let b=c.results.oct_12.scores[key]").replace("let b=c.results.current['8000_12000_signed_db']","let b=c.results.oct_12['8000_12000_signed_db']")
a=js.index("document.getElementById('status').textContent=");b=js.index(";document.getElementById('candidates')",a);js=js[:a]+"document.getElementById('status').textContent=group==='percussion'?'1792–1797초의 두 타격은 새 퍼커션 후보 설계에서 제외했습니다. 마지막 1/12 후보와 같은 위치에서 비교하세요.':'퍼커션에서 만든 고정 후보를 다른 주법에 그대로 적용한 비교입니다. 이 주법에서 새 후보를 설계하지 않았습니다.'"+js[b:]
a=js.index('function tables()');b=js.index("document.getElementById('groups').innerHTML=",a);js=js[:a]+"function tables(){}"+js[b:];js=js.replace("가운데: 앞선 후보 → 원본 보기","가운데: 마지막 1/12 → 원본 보기").replace("가운데: 원본 → 앞선 후보 보기","가운데: 원본 → 마지막 1/12 보기")
html=html[:start]+'<script>const data='+json.dumps(r,ensure_ascii=False,separators=(',',':'))+js+'</script>'+html[end+9:]
html=html.replace('<title>국소 중역 1/3·1/6·1/12옥타브 비교</title>','<title>퍼커션 지속 구간 기반 상대 응답 EQ</title>').replace('<h1>800Hz–1kHz · 1.5–2kHz를 더 세밀하게 보정</h1>','<h1>퍼커션 지속 구간 기반 보정 vs 마지막 1/12 필터</h1>')
a=html.index('<p class="lead">');b=html.index('</p>',a)+4;html=html[:a]+'<p class="lead">8개 타격의 실제 에너지 지속 구간에서 KM184와 기기의 상대 스펙트럼 차이를 구하고, 마지막 1/12 필터 위에 <b>퍼커션 기반 보완 하나</b>를 만들었습니다. 뒤의 5개 타격은 설계에서 제외했습니다. <b>마이크 IR이나 위상 복원이 아닌 상대 음색 EQ</b>이며, 화음·아르페지오로도 그대로 교차 적용합니다. <b>제외 타격은 개선됐지만 아르페지오·마지막 곡 예시는 악화되어, 공통 교체가 아닌 퍼커션용 비교 옵션입니다.</b></p>'+html[b:]
html=html.replace('앞선 밝기 보완','앞선 밝기 보완').replace('<nav>','<nav><a href="local-resolution-eq.html">마지막 1/12 비교</a>',1).replace('<th>앞선 후보</th>','<th>마지막 1/12</th>',1).replace('가운데: 앞선 후보 → 원본 보기','가운데: 마지막 1/12 → 원본 보기').replace('1/12 필터는 표시점만 늘린 것이 아니라 실제로 더 좁은 피킹 EQ를 사용합니다.','퍼커션 후보도 실제 1/12 폭의 피킹 EQ를 사용하며, 짧아 분해할 수 없는 저주파 관측은 제외합니다.')
a=html.index('<h2>다른 주법에서의 개발 확인</h2>');b=html.index('<script>',a)
fmt=lambda v:'미관측' if v is None else f'{v:.2f}'
rows=''
for d in ('iphone','s21'):
 for split,label in [('design','설계 8타격'),('held_hit','제외 5타격')]:
  for region,title in [('600_12000','600Hz–12kHz'),('800_2000','800Hz–2kHz'),('8000_12000','8–12kHz')]:
   z=results[d][split][region];rows+=f'<tr><td>{d.upper()}</td><td>{label}</td><td>{title}</td><td>{fmt(z["latest_mae_db"])}</td><td>{fmt(z["percussion_mae_db"])}</td><td>{z["worse_hits_gt_0_5db"]}/{z["hits"]}</td></tr>'
gates=''.join(f'<tr><td>{e["id"]}</td><td>{e["split"]}</td><td>{e["start_s"]:.3f}–{e["end_s"]:.3f}</td><td>{e["duration_s"]*1000:.0f}ms</td><td>{"상한 도달" if e["tail_cap"] else ""}</td></tr>' for e in saved)
cross=''.join(f'<tr><td>{c["device"].upper()} · {c["label"]}</td><td>{c["fullband_scores"]["oct_12"]:.2f}</td><td>{c["fullband_scores"]["percussion"]:.2f}</td></tr>' for c in r['clips'])
def measured_svg(d):
 model=models[d];cs=np.array(model['centers_hz']);med=np.array([np.nan if v is None else v for v in model['raw_median_residual_db']]);spread=np.array(model['mad_db']);valid=(np.array(model['support_hits'])>=4)&np.isfinite(med)
 x=lambda f:60+np.log(f/600)/np.log(12000/600)*735
 y=lambda db:30+(10-np.clip(db,-20,10))/30*220
 svg='<svg viewBox="0 0 850 280" role="img" aria-label="'+d+' 타격 상대 응답과 실제 보완 응답"><rect width="850" height="280" fill="white"/>'
 for v in (-20,-10,0,10):svg+=f'<line x1="60" x2="795" y1="{y(v)}" y2="{y(v)}" stroke="#d5dfe2"/><text x="10" y="{y(v)+4}" font-size="12">{v}dB</text>'
 for v in (600,1000,2000,4000,8000,12000):svg+=f'<text x="{x(v)}" y="270" text-anchor="middle" font-size="12">{v/1000:g}k</text>'
 for j in np.flatnonzero(valid):
  color='#7b969e' if model['reliability'][j]<.1 else '#c68b31'
  svg+=f'<line x1="{x(cs[j])}" x2="{x(cs[j])}" y1="{y(med[j]+spread[j])}" y2="{y(med[j]-spread[j])}" stroke="{color}"/><circle cx="{x(cs[j])}" cy="{y(med[j])}" r="2.5" fill="{color}"/>'
 hz=np.geomspace(600,12000,600);rv=response(np.array(model['extra_sos']),hz);points=' '.join(f'{x(fq):.2f},{y(db):.2f}' for fq,db in zip(hz,rv));svg+='<polyline points="'+points+'" fill="none" stroke="#ad428b" stroke-width="2.5"/></svg>'
 return '<div class="panel"><h3>'+d.upper()+'</h3>'+svg+'</div>'
measured_section='<h2>퍼커션에서 추정한 상대 FR와 실제 보완</h2><p>점은 설계 8타격에서 남은 KM184−마지막 후보의 중앙 잔차, 막대는 타격 간 MAD입니다. 양수는 부족해 증폭이 필요한 방향입니다. 회색은 낮은 신뢰도, 보라색은 실제 추가 필터 응답입니다. 관측 부족은 연결하지 않고, 반복성·퍼짐·±6dB 제한 때문에 필터가 관측 요철을 모두 따라가지는 않습니다.</p>'+''.join(measured_svg(d) for d in ('iphone','s21'))
content='<h2>설계에서 제외한 타격의 수치 비교</h2><p>각 타격의 1/12옥타브 대역 에너지 절대 오차(dB), 관측 신뢰도 가중. 출력 레벨은 설계 8타격의 200–600Hz에서 정한 <b>고정 게인 하나</b>를 유지합니다. 악화는 타격 평균 오차가 0.5dB 넘게 증가한 경우입니다. 앞선 필터는 이미 이 세션을 보았으므로 독립 검증은 아닙니다.</p><div class="panel"><table><thead><tr><th>기기</th><th>자료</th><th>대역</th><th>마지막 1/12</th><th>퍼커션 후보</th><th>악화 타격</th></tr></thead><tbody>'+rows+'</tbody></table></div><details><summary>다른 주법 5초 예시의 교차 적용</summary><p>600Hz–12kHz의 관측 가능 대역을 비교합니다. 이 표와 위 타격 표는 서로 다른 마스크·레벨 기준입니다. 예시마다 원본에서 정한 기존 기준 게인을 전후 동일하게 사용합니다. 개별 예시는 학습 제외 성능표가 아닙니다.</p><table><tr><th>기기·주법</th><th>마지막 1/12</th><th>퍼커션 후보</th></tr>'+cross+'</table></details><details><summary>타격별 시작·끝과 추정 해상도</summary><p>KM184 시작부터 세 기기 중 가장 늦게 관측되는 꼬리까지 같은 구간을 사용했습니다. −35dB/배경+6dB 아래 40ms 연속일 때 종료하고 최대 약0.83초를 제한했습니다. 실제 접촉 onset/offset의 수동 확정은 아닙니다. 분해능 조건은 대역폭 ≥3/지속시간입니다. FFT 빈을 늘려도 실제 해상도는 늘지 않습니다.</p><table><tr><th>타격</th><th>역할</th><th>KM184 시각</th><th>지속시간</th><th>주의</th></tr>'+gates+'</table></details><details><summary>설계 원칙과 계수</summary><p>주파수별 양쪽 SNR과 에너지 비중을 확인하고, 설계 8타격 중 최소 4타격에서 관측되는 성분을 사용합니다. 타격 간 방향 일치와 퍼짐에 따라 보정량을 줄입니다. 600Hz 이하 저역은 마지막 후보를 유지하며, 새 보완의 총 응답은 ±6dB로 제한했습니다. 실제 연속 문맥으로 필터링 후 중앙 5초를 자릅니다. 모든 버전의 표시 게인·안전 배율·스펙트로그램 색 범위가 같습니다. 고정 EQ는 공간·위상·감쇠 차이를 복원하지 못합니다.</p><a href="percussion-response-eq.json">타격 경계·마스크·잔차·SOS 계수 JSON</a></details>'
html=html[:a]+measured_section+content+html[b:];html=html.replace('그래프의 보완 응답은 앞선 후보와의 실제 차이입니다.','그래프의 보완 응답은 마지막 1/12 후보와의 실제 차이입니다.').replace('。','.')
(OUT/'percussion-response-eq.html').write_text(html+'\n');print(json.dumps(results,indent=2),flush=True)
