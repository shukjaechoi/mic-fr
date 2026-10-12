"""Reference-onset diagnostic; frozen 8-hit strength, 5-hit evaluation."""
from pathlib import Path
import json
import numpy as np
from scipy import signal,ndimage
import soundfile as sf
from PIL import Image
from matplotlib import colormaps
source=Path(__file__).with_name('reference_magnitude_diagnostic.py').read_text();exec(compile(source[:source.index('rows=[];grids=[];qa=[]')],str(__file__),'exec'))
diag=json.loads((OUT/'percussion-device-diagnosis.json').read_text());previous=json.loads((OUT/'reference-magnitude-browser.json').read_text());events=json.loads((OUT/'percussion-response-eq.json').read_text())['events'];bands=[(600,1200),(1200,2400),(2400,4800),(4800,8000),(8000,12000)]
def peak(hz,q,g):
 A=10**(g/40);w=2*np.pi*hz/FS;al=np.sin(w)/(2*q);c=np.cos(w);a=np.array([1+al/A,-2*c,1-al/A]);b=np.array([1+al*A,-2*c,1-al*A]);return np.r_[b/a[0],a/a[0]][None,:]
def envelope(n,left,onsets):
 t=np.arange(n)/FS+left;out=np.zeros(n)
 for on in onsets:
  dt=t-on;v=np.zeros(n);rise=(dt>=.04)&(dt<.07);hold=(dt>=.07)&(dt<.15);fall=(dt>=.15)&(dt<.3);v[rise]=.5-.5*np.cos(np.pi*(dt[rise]-.04)/.03);v[hold]=1;v[fall]=.5+.5*np.cos(np.pi*(dt[fall]-.15)/.15);out=np.maximum(out,v)
 return out
models={}
for d in ('iphone','s21'):
 tr=[r for r in diag['events'] if r['device']==d and r['split']=='design'];mid=[r['stage_residuals']['1200_2400']['body']['signed_device_minus_km_db'] for r in tr];hi=[r['stage_residuals']['8000_12000'][s]['signed_device_minus_km_db'] for r in tr for s in ('attack','body')];mg=-min(4,max(0,.5*np.median([v for v in mid if v is not None])));hg=-min(4,max(0,.5*np.median([v for v in hi if v is not None]))) if d=='s21' else 0
 models[d]={'mid_gain_db':float(mg),'high_gain_db':float(hg),'mid_sos':peak(np.sqrt(1200*2400),1.1,mg).tolist(),'high_sos':peak(10000,1.1,hg).tolist(),'fit_hit_ids':[r['id'] for r in tr]}
def process(d,a,b,onsets,anchor):
 raw=read(d,a,b);k=read('km184',a,b);gain=10**(anchor/20);base=read(d,a,b,sos=np.array(prior['models'][d]['oct_12']['sos']))*gain
 # Read baseline with another second before window to settle BOTH fixed paths.
 extended=read(d,a-1,b,sos=np.array(prior['models'][d]['oct_12']['sos']))*gain;high=signal.sosfilt(np.array(models[d]['high_sos']),extended);cut=signal.sosfilt(np.array(models[d]['mid_sos']),high);env=envelope(len(base),a,onsets);high=high[FS:];cut=cut[FS:];waves={'km184':k,'raw':raw*gain,'baseline':base,'new':high+env*(cut-high)}
 if d=='s21':waves['high']=high
 return waves,raw,env
# Zero-phase bandpass used only for SAME output evaluation, not processing.
def stage_scores(waves,raw,d,left,onsets):
 filt={key:[ndimage.uniform_filter1d(signal.sosfiltfilt(signal.butter(3,[lo,hi],btype='bandpass',fs=FS,output='sos'),v)**2,240) for lo,hi in bands] for key,v in waves.items()};rf=[ndimage.uniform_filter1d(signal.sosfiltfilt(signal.butter(3,[lo,hi],btype='bandpass',fs=FS,output='sos'),raw)**2,240) for lo,hi in bands];rows=[]
 for on in onsets:
  for st,lo,hi in [('attack',0,.04),('body',.04,.15),('decay',.15,.3)]:
   a=round((on+lo-left)*FS);b=round((on+hi-left)*FS)
   if a<0 or b>len(raw):continue
   for bi,(fl,fh) in enumerate(bands):
    m=(ff>=fl)&(ff<fh);ke=np.mean(filt['km184'][bi][a:b]);re=np.mean(rf[bi][a:b]);sn=min(10*np.log10(max(ke/(np.sum(floors['km184'][m])*(ff[1]-ff[0])),1e-30)),10*np.log10(max(re/(np.sum(floors[d][m])*(ff[1]-ff[0])),1e-30)))
    if sn<10:continue
    rows.append({'onset_s':on,'stage':st,'band_hz':[fl,fh],'joint_raw_snr_db':float(sn),'signed_db':{key:float(10*np.log10(max(np.mean(v[bi][a:b])/ke,1e-30))) for key,v in filt.items() if key!='km184'}})
 return rows
held=[]
for d in models:
 anchor=next(c['anchor_db'] for c in prior['clips'] if c['device']==d and c['group']=='percussion')
 for e in events:
  a=e['start_s'];w,raw,env=process(d,a-.5,a+.4,[a],anchor);held.append({'device':d,'id':e['id'],'split':e['split'],'start_s':a,'stages':stage_scores(w,raw,d,a-.5,[a])})
clips=[];qa=[]
for c in previous['clips']:
 d=c['device'];a=c['start_s'];ons=[e['start_s'] for e in events if a-.3<e['start_s']<a+5] if c['group']=='percussion' else sorted(set(a+r['onset_clip_s'] for r in c['stages']['events']));w,raw,env=process(d,a-.5,a+5.5,ons,c['anchor_db']);stages=stage_scores(w,raw,d,a-.5,ons);crop=slice(FS//2,FS//2+5*FS);w={k:v[crop] for k,v in w.items()};raw=raw[crop]
 # Previous reference-assisted time correction for direct listening; undo its shared safety scale.
 oldwave,_=sf.read(OUT/c['media']['temporal']['audio']);w['temporal']=oldwave/c['common_safety_scale'];wt=weights(powers(stft(w['km184'])[2]),powers(stft(raw)[2]),d)[0];wt*=wt>=.15;scores={k:metrics(v,w['km184'],raw,d,wt) for k,v in w.items() if k!='km184'}
 tp=max(np.max(abs(signal.resample_poly(v,4,1))) for v in w.values());scale=min(1,.8/tp);w={k:v*scale for k,v in w.items()};specs={k:image_spec(v) for k,v in w.items()};top=max(v.max() for v in specs.values());media={}
 for key,v in w.items():
  stem=f'transient-{d}-{c["group"]}-{key}';sf.write(OUT/(stem+'.wav'),v,FS,subtype='PCM_16');Image.fromarray(np.uint8(colormaps['magma'](np.clip((specs[key]-top+72)/72,0,1))[:,:,:3]*255)).save(OUT/(stem+'.png'));media[key]={'audio':stem+'.wav','spectrogram':stem+'.png'}
  if key!='km184':
   delta=specs[key]-specs['km184'];rgb=colormaps['RdBu_r'](np.clip((delta+12)/24,0,1))[:,:,:3];rgb[(specs[key]<top-58)|(specs['km184']<top-58)]=[.88,.9,.91];Image.fromarray(np.uint8(rgb*255)).save(OUT/(stem+'-error.png'));media[key]['residual']=stem+'-error.png'
 clips.append({'device':d,'group':c['group'],'label':c['label'],'start_s':a,'onsets_clip_s':[o-a for o in ons],'anchor_db':c['anchor_db'],'media':media,'scores':scores,'stages':stages,'safety_scale':float(scale),'true_peak_dbfs':float(20*np.log10(tp*scale))});qa.append({'device':d,'group':c['group'],'true_peak_dbfs':float(20*np.log10(tp*scale)),'finite':all(np.all(np.isfinite(v)) for v in w.values()),'mid_envelope_max_step':float(np.max(abs(np.diff(env))))});print(d,c['group'],{k:round(v['frame_mae_db'],3) for k,v in scores.items()},flush=True)
summary={}
for d in models:
 rr=[r for h in held if h['device']==d and h['split']=='held_hit' for r in h['stages']];summary[d]={}
 for band in bands:
  summary[d][str(band)]={}
  for st in ('attack','body','decay'):
   rows=[r for r in rr if r['band_hz']==list(band) and r['stage']==st];summary[d][str(band)][st]={key:{'median_signed_db':float(np.median([r['signed_db'][key] for r in rows])) if rows else None,'mae_db':float(np.mean([abs(r['signed_db'][key]) for r in rows])) if rows else None,'observed_hits':len(rows)} for key in ('raw','baseline',*(['high'] if d=='s21' else []),'new')}
fgrid=np.geomspace(80,18000,300);responses={}
for d,m in models.items():
 _,hm=signal.sosfreqz(np.array(m['mid_sos']),worN=fgrid,fs=FS);_,hh=signal.sosfreqz(np.array(m['high_sos']),worN=fgrid,fs=FS);responses[d]={'frequency_hz':fgrid.tolist(),'high_db':(20*np.log10(abs(hh))).tolist(),'full_body_db':(20*np.log10(abs(hh*hm))).tolist(),'half_body_db':(20*np.log10(abs(hh*(.5+.5*hm)))).tolist(),'pole_radius_max':float(max(abs(np.roots(np.array(m['mid_sos'])[0,3:]))))}
r={'scope':'Known-reference-onset diagnostic. First8 strength fit/final5 fixed developmental evaluation; explored same session. Cross-technique uses KM184 energy candidates, not autonomous input-only detector. Existing anchors remain paired-derived. Previous temporal audio is same-clip reference-assisted diagnostic.','rules':Path(__file__).with_name('TRANSIENT_STAGE_PLAN.md').read_text(),'models':models,'responses':responses,'held_summary':summary,'hit_results':held,'clips':clips,'qa':qa};(OUT/'transient-stage-filter.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps({'models':models,'held_summary':summary},ensure_ascii=False))
