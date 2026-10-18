"""Direct magnitude / cross-phase FIR diagnostic, frozen first8 fit/final5 eval."""
from pathlib import Path
import json,hashlib
import numpy as np
from scipy import signal,ndimage
import soundfile as sf
from PIL import Image
from matplotlib import colormaps
s=Path(__file__).with_name('reference_magnitude_diagnostic.py').read_text();exec(compile(s[:s.index('rows=[];grids=[];qa=[]')],str(__file__),'exec'))
SRC=Path('/Users/sjchoi/dev/rec/261002-iphone,h1,km184,s21,d10');paths={'km184':SRC/'neumann km184 guitar.wav','s21':SRC/'s21 guitar.wav','iphone':Path('/private/tmp/mic-audit-20261008/iphone-decoded-stereo.wav')};events=json.loads((OUT/'percussion-response-eq.json').read_text())['events'];N=32768;TAPS=4097;MID=TAPS//2;freqfit=np.fft.rfftfreq(N,1/FS);selected={};offsets={};models={};qc={}
original_read=read
def readst(d,a,b,channel='mono'):
 st=mapped(d,a);en=mapped(d,b)
 with sf.SoundFile(paths[d]) as f:f.seek(st);v=f.read(en-st,always_2d=True)
 v=v.mean(axis=1) if channel=='mono' else v[:,int(channel)]
 count=round((b-a)*FS);return signal.resample(v,count) if len(v)!=count else v
hp=signal.butter(3,[300,10000],btype='bandpass',fs=FS,output='sos')
def lagcor(k,x):
 a=signal.sosfiltfilt(hp,k);b=signal.sosfiltfilt(hp,x);cc=signal.correlate(a,b,method='fft');lags=signal.correlation_lags(len(a),len(b));take=abs(lags)<=240;inds=np.flatnonzero(take);j=inds[np.argmax(abs(cc[take]))];lag=float(lags[j]);
 if 0<j<len(cc)-1:
  z=abs(cc[j-1:j+2]);den=z[0]-2*z[1]+z[2]
  if abs(den)>1e-20:lag+=.5*(z[0]-z[2])/den
 return -lag/FS,float(cc[j]/np.sqrt(np.dot(a,a)*np.dot(b,b)))
for d in ('iphone','s21'):
 chans=['mono']+list(range(sf.info(paths[d]).channels));cr={str(ch):[] for ch in chans};lags={str(ch):[] for ch in chans};att=[]
 for e in events[:8]:
  a=e['start_s'];k=readst('km184',a-.08,a+.32)
  xx=[readst(d,a-.08,a+.32,ch) for ch in range(sf.info(paths[d]).channels)];att.append(float(20*np.log10(np.sqrt(np.mean(np.mean(xx,axis=0)**2))/np.sqrt(np.mean(np.array(xx)**2)))))
  for ch in chans:
   delta,c=lagcor(k,readst(d,a-.08,a+.32,ch));cr[str(ch)].append(c);lags[str(ch)].append(delta)
 best=max(chans,key=lambda ch:np.median(np.abs(cr[str(ch)])));selected[d]=best if np.median(np.abs(cr[str(best)]))>=np.median(np.abs(cr['mono']))+.05 else 'mono';offsets[d]=float(np.median(lags[str(selected[d])]))
 qc[d]={'selected_channel':selected[d],'channel_median_abs_correlation':{ch:float(np.median(abs(np.array(v)))) for ch,v in cr.items()},'train_offset_ms':[1000*v for v in lags[str(selected[d])]],'fixed_offset_ms':1000*offsets[d],'mono_attenuation_db':float(np.median(att))};print('channel/lag',d,qc[d],flush=True)
def source(d,a,b):return readst(d,a+offsets[d],b+offsets[d],selected[d])
for d in ('iphone','s21'):
 # Fixed train level: sum of physical band energy over8; no per-eval fitting.
 pks=[];pxs=[]
 for e in events[:8]:
  a=e['start_s'];k=readst('km184',a-.08,a+.32);x=source(d,a-.08,a+.32);f,pk=signal.periodogram(k,FS,nfft=N);px=signal.periodogram(x,FS,nfft=N)[1];take=(f>=200)&(f<600);pks.append(np.sum(pk[take]));pxs.append(np.sum(px[take]))
 anchor=float(10*np.log10(np.sum(pks)/np.sum(pxs)));gain=10**(anchor/20);xx=[];yy=[];invrms=[]
 for e in events[:8]:
  a=e['start_s'];k=readst('km184',a-.08,a+.32);x=source(d,a-.08,a+.32)*gain;rms=np.sqrt(np.mean(x*x));win=signal.windows.tukey(len(x),.1);xx.append(np.fft.rfft(x/rms*win,n=N));yy.append(np.fft.rfft(k/rms*win,n=N));invrms.append(1/rms**2)
 X=np.array(xx);Y=np.array(yy);sm=lambda v:ndimage.gaussian_filter1d(v,3);px=sm(np.mean(abs(X)**2,axis=0));py=sm(np.mean(abs(Y)**2,axis=0));xy=np.mean(X.conj()*Y,axis=0);xy=sm(xy.real)+1j*sm(xy.imag);coh=np.clip(abs(xy)**2/np.maximum(px*py,1e-30),0,1)
 nf=np.interp(freqfit,ff,floors[d])*gain**2*FS*np.sum(win**2)/2*np.mean(invrms);nk=np.interp(freqfit,ff,floors['km184'])*FS*np.sum(win**2)/2*np.mean(invrms);snr=np.minimum(10*np.log10(np.maximum(px/np.maximum(nf,1e-30),1e-30)),10*np.log10(np.maximum(py/np.maximum(nk,1e-30),1e-30)));conf=np.clip((snr-10)/15,0,1);taper=np.minimum(np.clip((freqfit-150)/150,0,1),np.clip((14000-freqfit)/2000,0,1));rel=conf*taper
 # Realized-complex magnitude is re-synthesized as magnitude-only control, to isolate phase; lower-coherence phase shrinks toward identity.
 hreg=xy/np.maximum(px*1.05+nf,1e-30);db=np.clip(20*np.log10(np.maximum(abs(hreg),1e-12)),-12,12)*rel;mag=10**(db/20);pw=np.clip((coh-.2)/.6,0,1)*rel;vec=(1-pw)+pw*np.exp(1j*np.angle(hreg));phase=vec/np.maximum(abs(vec),1e-12);phase[abs(vec)<1e-8]=1;hc=mag*phase
 def realize(h):
  imp=np.fft.fftshift(np.fft.irfft(h,n=N));return imp[N//2-MID:N//2+MID+1]*np.kaiser(TAPS,6)
 complex_fir=realize(hc);matched_magnitude=abs(np.fft.rfft(complex_fir,n=N));firs={'magnitude':realize(matched_magnitude),'complex':complex_fir};hz=np.geomspace(80,18000,400);responses={}
 for key,h in firs.items():
  _,hr=signal.freqz(h,worN=hz,fs=FS);hr*=np.exp(2j*np.pi*hz*MID/FS);responses[key]={'magnitude_db':(20*np.log10(np.maximum(abs(hr),1e-12))).tolist(),'phase_rad':np.unwrap(np.angle(hr)).tolist()};sf.write(OUT/f'complex-linear-{d}-{key}-fir.wav',h,FS,subtype='FLOAT')
 models[d]={'anchor_db':anchor,'offset_ms':offsets[d]*1000,'selected_channel':selected[d],'fir':{k:v.tolist() for k,v in firs.items()},'fir_sha256':{k:hashlib.sha256(v.tobytes()).hexdigest() for k,v in firs.items()},'response_hz':hz.tolist(),'responses':responses,'coherence_frequency_hz':freqfit[::16].tolist(),'coherence':coh[::16].tolist(),'observed_fraction_600_12000':float(np.mean(conf[(freqfit>=600)&(freqfit<=12000)]>=.15)),'realized_response_difference_rms_db':float(np.sqrt(np.mean((np.array(responses['magnitude']['magnitude_db'])-np.array(responses['complex']['magnitude_db']))**2))),'rules':'Magnitude control re-synthesizes realized complex-FIR magnitude, same4097length. Residual truncation mismatch reported; phase uses coherence .2–.8 identity shrink.'};print('fit',d,anchor,models[d]['realized_response_difference_rms_db'],flush=True)
legacy=json.loads((OUT/'fft-order-eq.json').read_text())
def process(d,a,b):
 left=a-.5;right=b+.5;x=source(d,left,right);gain=10**(models[d]['anchor_db']/20);base=source(d,left-1,right);base=signal.sosfilt(np.array(prior['models'][d]['oct_12']['sos']),base)[FS:];crop=slice(FS//2,FS//2+round((b-a)*FS));w={'km184':readst('km184',a,b),'raw':x[crop]*gain,'baseline':base[crop]*gain}
 old=legacy['models'][d]['fir1025'];oldinput=signal.sosfilt(np.array(old['base_sos']),source(d,left-1,right))[FS:];w['previous']=signal.fftconvolve(oldinput,np.array(old['fir']),mode='same')[crop]*gain
 for key,h in models[d]['fir'].items():w[key]=signal.fftconvolve(x,np.array(h),mode='same')[crop]*gain
 for key,trim in models[d].get('comparison_train_trim_db',{}).items():w[key]*=10**(trim/20)
 return w,x[crop]
def measures(w,raw,d):
 wt=weights(powers(stft(w['km184'])[2]),powers(stft(raw)[2]),d)[0];wt*=wt>=.15;out={}
 for key,v in w.items():
  if key=='km184':continue
  out[key]=metrics(v,w['km184'],raw,d,wt);out[key]['waveform_nmse_db']=float(10*np.log10(max(np.sum((v-w['km184'])**2)/np.sum(w['km184']**2),1e-30)));out[key]['spectral_convergence']={}
  for L2 in (960,4096):
   zk=signal.stft(w['km184'],FS,nperseg=L2,noverlap=L2*3//4)[2];zz=signal.stft(v,FS,nperseg=L2,noverlap=L2*3//4)[2];out[key]['spectral_convergence'][str(L2)]=float(np.linalg.norm(abs(zz)-abs(zk))/np.linalg.norm(abs(zk)))
 return out
# Existing complete profiles carry their own low-band gain. Calibrate their
# output once on design8 using SAME200–600 criterion; never fit evaluation gain.
for d in models:
 sums={k:[] for k in ('km184','baseline','previous')}
 for e in events[:8]:
  a=e['start_s'];w,_=process(d,a-.08,a+.32)
  for key in sums:
   ff2,p=signal.periodogram(w[key],FS,nfft=N);sums[key].append(float(np.sum(p[(ff2>=200)&(ff2<600)])))
 models[d]['comparison_train_trim_db']={k:float(10*np.log10(np.sum(sums['km184'])/np.sum(sums[k]))) for k in ('baseline','previous')}
 print('legacy training output trims',d,models[d]['comparison_train_trim_db'],flush=True)
stage_source=Path(__file__).with_name('transient_stage_filter.py').read_text();bands=[(600,1200),(1200,2400),(2400,4800),(4800,8000),(8000,12000)];exec(compile(stage_source[stage_source.index('def stage_scores('):stage_source.index('held=[]')],str(__file__),'exec'))
hitrows=[]
for d in models:
 for e in events:
  a=e['start_s'];w,raw=process(d,a-.08,a+.32);delta,corr=lagcor(w['km184'],raw);hitrows.append({'device':d,'id':e['id'],'split':e['split'],'start_s':a,'residual_lag_ms':delta*1000,'abs_correlation':abs(corr),'scores':measures(w,raw,d),'stages':stage_scores(w,raw,d,a-.08,[a])})
clips=[]
for d in models:
 for group,a,label in [('percussion',1792,'퍼커션'),('complex',1704,'복합 화음'),('arpeggio',100,'아르페지오')]:
  w,raw=process(d,a,a+5);prevclip=next(c for c in json.loads((OUT/'reference-magnitude-browser.json').read_text())['clips'] if c['device']==d and c['group']==group);onsets=sorted(set(a+e['onset_clip_s'] for e in prevclip['stages']['events']));stages=stage_scores(w,raw,d,a,onsets);scores=measures(w,raw,d);tp=max(np.max(abs(signal.resample_poly(v,4,1))) for v in w.values());scale=min(1,.8/tp);w={k:v*scale for k,v in w.items()};sp={k:image_spec(v) for k,v in w.items()};top=max(v.max() for v in sp.values());media={}
  for key,v in w.items():
   stem=f'complex-linear-{d}-{group}-{key}';sf.write(OUT/(stem+'.wav'),v,FS,subtype='PCM_16');Image.fromarray(np.uint8(colormaps['magma'](np.clip((sp[key]-top+72)/72,0,1))[:,:,:3]*255)).save(OUT/(stem+'.png'));media[key]={'audio':stem+'.wav','spectrogram':stem+'.png'}
   if key!='km184':
    delta=sp[key]-sp['km184'];rgb=colormaps['RdBu_r'](np.clip((delta+12)/24,0,1))[:,:,:3];rgb[(sp[key]<top-58)|(sp['km184']<top-58)]=[.88,.9,.91];Image.fromarray(np.uint8(rgb*255)).save(OUT/(stem+'-error.png'));media[key]['residual']=stem+'-error.png'
  clips.append({'device':d,'group':group,'label':label,'start_s':a,'scores':scores,'stages':stages,'media':media,'safety_scale':scale,'true_peak_dbfs':float(20*np.log10(tp*scale))});print('clip',d,group,{k:round(v['frame_mae_db'],3) for k,v in scores.items()},flush=True)
summary={}
for d in models:
 rr=[h for h in hitrows if h['device']==d and h['split']=='held_hit'];summary[d]={key:{metric:float(np.mean([h['scores'][key][metric] for h in rr])) for metric in ('frame_mae_db','waveform_nmse_db')} for key in ('raw','baseline','previous','magnitude','complex')}
r={'scope':'Direct raw-input linear FIR development. First8 train /final5 previously explored hits. Fixed train-derived channel/time offset/gain, no evaluation-specific alignment. Not independent validation or deployable realtime filter. Known source placement differences remain.','rules':Path(__file__).with_name('COMPLEX_LINEAR_PLAN.md').read_text(),'channel_alignment':qc,'models':models,'held_summary':summary,'hit_results':hitrows,'clips':clips,'taps':TAPS,'delay_ms':MID/FS*1000};(OUT/'complex-linear-percussion.json').write_text(json.dumps(r,ensure_ascii=False,indent=2,allow_nan=False));print('SUMMARY',json.dumps(summary),flush=True)
