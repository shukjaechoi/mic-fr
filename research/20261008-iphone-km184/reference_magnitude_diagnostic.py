"""Bounded, confidence-weighted oracle magnitude diagnostic; not deployable EQ."""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy import signal, ndimage
from PIL import Image
from matplotlib import colormaps
import soundfile as sf
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'docs/experiments/20261008-iphone-km184-public';LOCAL=ROOT/'docs/experiments/20261008-iphone-km184';FS=48000;L=4096;H=512
prior=json.loads((OUT/'local-resolution-eq-browser.json').read_text());old=json.loads((ROOT/'docs/experiments/20261002/data/analysis.json').read_text());bg=json.loads((LOCAL/'guitar-background-v2.json').read_text())
maps={m['id']:m['guitarSync'] for m in old['microphones']};maps['iphone']=json.loads((LOCAL/'analysis.json').read_text())['sets']['guitar']['mapping'];audio={d:np.load(f'/private/tmp/mic-audit-20261008/{d}-guitar.npy',mmap_mode='r') for d in ('km184','iphone','s21')}
def mapped(d,t):
 m=maps[d];return round(t*FS*m['slope']+m.get('interceptSamples',m.get('intercept_samples',0)))
def read(d,a,b,sos=None):
 st=mapped(d,a);en=mapped(d,b);left=max(0,st-FS);v=np.asarray(audio[d][left:en],float)
 if sos is not None:v=signal.sosfilt(sos,v)
 v=v[st-left:];return signal.resample(v,round((b-a)*FS)) if len(v)!=round((b-a)*FS) else v
ff=np.fft.rfftfreq(8192,1/FS);floors={d:np.array(bg['median_psd'][d]) for d in ('km184','iphone')};floors['s21']=np.median([signal.welch(read('s21',q['start_s'],q['start_s']+2),FS,nperseg=8192)[1] for q in bg['selected']],axis=0)
cs=1000*2**(np.arange(-5,23)/6);cs=cs[(cs>=600)&(cs<=12000)];freq=np.fft.rfftfreq(L,1/FS);masks=[(freq>=c/2**(1/12))&(freq<c*2**(1/12)) for c in cs]
noise={d:np.array([np.mean(np.interp(freq,ff,p)[m]) for m in masks]) for d,p in floors.items()}
def stft(x):return signal.stft(x,FS,nperseg=L,noverlap=L-H,boundary='zeros',padded=True)
def powers(z):return abs(z)**2/(FS*np.sum(signal.windows.hann(L,sym=False)**2)/np.sum(signal.windows.hann(L,sym=False))**2)*2
# The factor above equals 2*sum(window)^2/(fs*sum(window^2)); it yields one-sided PSD.
def bp(p):return np.array([np.mean(p[m],axis=0) for m in masks])
def weights(pk,pr,d):
 k,r=bp(pk),bp(pr);snr=np.minimum(10*np.log10(np.maximum(k/noise['km184'][:,None],1e-30)),10*np.log10(np.maximum(r/noise[d][:,None],1e-30)))
 # Mean PSD times bandwidth; octave-like energy ranking, not peak PSD ranking.
 ee=lambda p:np.max(np.array([np.sum(p[(freq>=lo)&(freq<hi)],axis=0)*(freq[1]-freq[0]) for lo,hi in [(100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6400),(6400,12000)]]),axis=0)
 width=cs*(2**(1/12)-2**(-1/12));rel=np.minimum(10*np.log10(np.maximum(k*width[:,None]/np.maximum(ee(pk),1e-30),1e-30)),10*np.log10(np.maximum(r*width[:,None]/np.maximum(ee(pr),1e-30),1e-30)))
 return np.clip((snr-10)/15,0,1)*(.2+.8*np.clip((rel+36)/24,0,1)),snr

def smooth_obs(v,w):
 numerator=ndimage.gaussian_filter1d(v*w,.6,axis=0,mode='nearest');den=ndimage.gaussian_filter1d(w,.6,axis=0,mode='nearest')
 return np.clip(numerator/np.maximum(den,1e-12),-6,6)*np.clip(den/.5,0,1)
def expand(g):
 out=np.array([np.interp(freq,cs,g[:,i],left=g[0,i],right=g[-1,i]) for i in range(g.shape[1])]).T
 taper=np.minimum(np.clip((freq-500)/100,0,1),np.clip((14000-freq)/2000,0,1));return out*taper[:,None]
def inv(z,n):return signal.istft(z,FS,nperseg=L,noverlap=L-H,boundary=True)[1][:n]
def weighted(v,w):return float(np.sum(v*w)/np.sum(w)) if np.sum(w)>0 else None
regions={'600_12000':(600,12000),'800_1000':(780,1030),'1500_2000':(1450,2050),'8000_12000':(7800,12000)}
def metrics(x,k,raw,d,wt):
 _,tt,z=stft(x);p=powers(z);kp=powers(stft(k)[2]);err=10*np.log10(np.maximum(bp(p),1e-30)/np.maximum(bp(kp),1e-30));whole={}
 # Pool energy across time, then score same original-mask confidence for each band.
 ep=10*np.log10(np.maximum(np.mean(bp(p),axis=1),1e-30)/np.maximum(np.mean(bp(kp),axis=1),1e-30));wg=np.mean(wt,axis=1)
 for name,(lo,hi) in regions.items():
  take=(cs>=lo)&(cs<=hi);w=wg*take;whole[name]={'mae_db':weighted(abs(ep),w),'signed_device_minus_km_db':weighted(ep,w),'mean_confidence':float(np.mean(wg[take])),'observed_bands':int(np.sum(wg[take]>=.15))}
 return {'pooled':whole,'frame_mae_db':weighted(abs(err),wt),'band_residual_db':ep.tolist()}
def phase_metrics(waves,raw,d,group,start):
 # KM184-envelope event detection, with non-overlapping 300ms windows only.
 hp=signal.butter(3,[600,12000],btype='bandpass',fs=FS,output='sos');k=signal.sosfiltfilt(hp,waves['km184']);env=ndimage.uniform_filter1d(k*k,480);pks,_=signal.find_peaks(env,distance=round(.3*FS),prominence=max(env)*.08,height=max(env)*.15)
 if group=='percussion':
  saved=json.loads((OUT/'percussion-response-eq.json').read_text())['events'];pks=np.array([round((e['start_s']-start)*FS) for e in saved if start+.06<e['start_s']<start+4.7])
 records=[];groups={stage:{key:[] for key in ('baseline','static','temporal')} for stage in ('attack','body','decay')}
 for pi in pks:
  if pi<.06*FS or pi>.7*FS+4*FS:continue
  # Rise threshold within60ms; energy onset rather than verified contact onset.
  left=max(0,pi-2880);on=np.flatnonzero(env[left:pi+1]>=env[pi]*.1);on=int(pi) if group=='percussion' else left+int(on[0])
  for stage,lo,hi in [('attack',0,.04),('body',.04,.15),('decay',.15,.3)]:
   a=on+round(lo*FS);b=on+round(hi*FS)
   if b>len(k) or any(on<q< b for q in pks if q>pi):continue
   pp={key:signal.periodogram(v[a:b],FS,nfft=8192,window='hann')[1] for key,v in waves.items()};rawp=signal.periodogram(raw[a:b],FS,nfft=8192,window='hann')[1];take=(ff>=600)&(ff<12000);ke=np.sum(pp['km184'][take]);re=np.sum(rawp[take]);sn=min(10*np.log10(ke/np.sum(floors['km184'][take])),10*np.log10(re/np.sum(floors[d][take])))
   if sn<10:continue
   rr={'onset_clip_s':on/FS,'stage':stage,'joint_snr_db':float(sn),'signed':{}}
   for key in ('baseline','static','temporal'):
    v=float(10*np.log10(np.sum(pp[key][take])/ke));groups[stage][key].append(v);rr['signed'][key]=v
   records.append(rr)
 return {'events':records,'stage_summary':{stage:{key:{'mae_db':float(np.mean(abs(np.array(v)))) if v else None,'signed_db':float(np.mean(v)) if v else None,'observations':len(v)} for key,v in gg.items()} for stage,gg in groups.items()}}
def peakcontrast(waves,raw,d):
 ps={key:signal.welch(v,FS,nperseg=8192,noverlap=4096)[1] for key,v in waves.items()};kp=10*np.log10(np.maximum(ps['km184'],1e-30));peaks,_=signal.find_peaks(kp,prominence=6,distance=8);out=[];rp=signal.welch(raw,FS,nperseg=8192,noverlap=4096)[1]
 for j in peaks:
  hz=ff[j]
  if not 700<hz<6000:continue
  core=abs(ff-hz)<=20;flank=(abs(ff-hz)>=45)&(abs(ff-hz)<=100)
  if min(10*np.log10(np.mean(ps['km184'][m])/np.mean(floors['km184'][m])) for m in (core,flank))<10 or min(10*np.log10(np.mean(rp[m])/np.mean(floors[d][m])) for m in (core,flank))<10:continue
  vals={key:float(10*np.log10(np.mean(v[core])/np.mean(v[flank]))) for key,v in ps.items()}
  out.append({'frequency_hz':float(hz),'contrast_db':vals,'baseline_difference_db':vals['baseline']-vals['km184']})
 out=sorted(out,key=lambda x:abs(x['baseline_difference_db']),reverse=True)[:8]
 return {'definition':'KM184-selected local spectral peaks, core ±20Hz versus flanks45–100Hz. Not automatically verified harmonics. Both raw core and flank SNR>=10dB.','peaks':out,'contrast_mae_db':{key:float(np.mean([abs(r['contrast_db'][key]-r['contrast_db']['km184']) for r in out])) if out else None for key in ('baseline','static','temporal')}}
freqgrid=np.geomspace(12000,80,320)
def image_spec(x):
 f,t,z=signal.stft(x,FS,nperseg=4096,noverlap=3584,boundary=None);fi,ti=np.meshgrid(freqgrid/(f[1]-f[0]),(np.linspace(0,5,470)-t[0])/(t[1]-t[0]),indexing='ij');return ndimage.map_coordinates(20*np.log10(np.maximum(abs(z),1e-9)),[fi,ti],order=1,mode='nearest')
rows=[];grids=[];qa=[]
for clip in prior['clips']:
 if clip['group'] not in ('arpeggio','complex','percussion'):continue
 d=clip['device'];a=clip['start_s'];anchor=clip['anchor_db'];sos=np.array(prior['models'][d]['oct_12']['sos']);left=a-.5;right=a+5.5;n=round(6*FS);crop=slice(24000,264000)
 assert not (a<1767.403 and a+5>1759.524)
 raw=read(d,left,right);ref=read('km184',left,right);base=read(d,left,right,sos=sos)*10**(anchor/20);gain=10**(anchor/20)
 _,tt,z=stft(base);zk=stft(ref)[2];zr=stft(raw)[2];pk,pr,pb=powers(zk),powers(zr),powers(z);wt,snr=weights(pk,pr,d);central=(tt>=.5)&(tt<5.5);err=10*np.log10(np.maximum(bp(pk),1e-30)/np.maximum(bp(pb),1e-30));observed=wt>=.15;wt=wt*observed
 # Static fit uses pooled physical energy with the same central-frame original support.
 wc=wt[:,central];support=np.sum(wc>0,axis=1);wglobal=np.sum(wc*wc,axis=1)/np.maximum(np.sum(wc,axis=1),1e-12);wglobal[support<3]=0;static_err=10*np.log10(np.maximum(np.sum(bp(pk)[:,central]*wc,axis=1),1e-30)/np.maximum(np.sum(bp(pb)[:,central]*wc,axis=1),1e-30));sg=smooth_obs(static_err[:,None],wglobal[:,None]);sg=np.repeat(sg,len(tt),axis=1)
 tg=smooth_obs(err,wt);num=ndimage.gaussian_filter1d(tg*wt,.06/(H/FS),axis=1,mode='nearest');den=ndimage.gaussian_filter1d(wt,.06/(H/FS),axis=1,mode='nearest');tg=np.clip(num/np.maximum(den,1e-12),-6,6)*np.clip(den/.5,0,1)
 # Two-sided rate bound is offline; time-varying smoothing does not imply causal implementation.
 for i in range(1,tg.shape[1]):tg[:,i]=np.clip(tg[:,i],tg[:,i-1]-24*H/FS,tg[:,i-1]+24*H/FS)
 for i in range(tg.shape[1]-2,-1,-1):tg[:,i]=np.clip(tg[:,i],tg[:,i+1]-24*H/FS,tg[:,i+1]+24*H/FS)
 recon=inv(z,n);waves={'km184':ref[crop],'raw':raw[crop]*gain,'baseline':base[crop],'static':inv(z*10**(expand(sg)/20),n)[crop],'temporal':inv(z*10**(expand(tg)/20),n)[crop]}
 assert max(abs(recon[crop]-base[crop]))<1e-10
 rawclip=raw[crop];wtclip=weights(powers(stft(waves['km184'])[2]),powers(stft(rawclip)[2]),d)[0];wtclip*=wtclip>=.15
 scores={key:metrics(v,waves['km184'],rawclip,d,wtclip) for key,v in waves.items() if key!='km184'};phases=phase_metrics(waves,rawclip,d,clip['group'],a);peaks=peakcontrast(waves,rawclip,d)
 # Residual alignment clue only: compare broadband energy envelopes after mapping.
 e1=ndimage.uniform_filter1d(waves['km184']**2,960)[::240];e2=ndimage.uniform_filter1d(waves['raw']**2,960)[::240];lags=signal.correlation_lags(len(e1),len(e2));cc=signal.correlate(e1-e1.mean(),e2-e2.mean());take=abs(lags)<=20;lag=float(lags[take][np.argmax(cc[take])]*5)
 maxsample=max(np.max(abs(v)) for v in waves.values());maxtrue=max(np.max(abs(signal.resample_poly(v,4,1))) for v in waves.values());scale=min(1,.8/max(maxsample,maxtrue));waves={key:v*scale for key,v in waves.items()};assert all(np.all(np.isfinite(v)) for v in waves.values())
 med={};specs={key:image_spec(v) for key,v in waves.items()};top=max(s.max() for s in specs.values());refsp=specs['km184']
 for key,v in waves.items():
  stem=f'refdiag-{d}-{clip["group"]}-{key}';sf.write(OUT/(stem+'.wav'),v,FS,subtype='PCM_16');Image.fromarray(np.uint8(colormaps['magma'](np.clip((specs[key]-top+72)/72,0,1))[:,:,:3]*255)).save(OUT/(stem+'.png'),optimize=True);med[key]={'audio':stem+'.wav','spectrogram':stem+'.png'}
  if key!='km184':
   delta=specs[key]-refsp;rgb=colormaps['RdBu_r'](np.clip((delta+12)/24,0,1))[:,:,:3];rgb[(refsp<top-58)|(specs[key]<top-58)]=[.88,.9,.91];Image.fromarray(np.uint8(rgb*255)).save(OUT/(stem+'-error.png'),optimize=True);med[key]['residual']=stem+'-error.png'
 for key,g in [('static',sg),('temporal',tg)]:
  # Display gain on log-frequency vs common clip time, with explicit ±6 scale.
  fi=freqgrid/(FS/L);ti=np.interp(np.linspace(.5,5.5,470),tt,np.arange(len(tt)));fg,tgix=np.meshgrid(fi,ti,indexing='ij');mat=ndimage.map_coordinates(expand(g),[fg,tgix],order=1,mode='nearest');name=f'refdiag-{d}-{clip["group"]}-{key}-gain.png';Image.fromarray(np.uint8(colormaps['RdBu_r']((mat+6)/12)[:,:,:3]*255)).save(OUT/name,optimize=True);med[key]['gain']=name
 name=f'refdiag-{d}-{clip["group"]}-confidence.png';fi=np.interp(freqgrid,cs,np.arange(len(cs)),left=0,right=len(cs)-1);ti=np.interp(np.linspace(.5,5.5,470),tt,np.arange(len(tt)));fg,tgix=np.meshgrid(fi,ti,indexing='ij');cm=ndimage.map_coordinates(wt,[fg,tgix],order=1,mode='nearest');cm[(freqgrid<cs[0])|(freqgrid>cs[-1])]=0;Image.fromarray(np.uint8(colormaps['viridis'](cm)[:,:,:3]*255)).save(OUT/name,optimize=True)
 observation={'confidence_image':name,'observed_time_band_fraction':float(np.mean(wt[:,central]>0)),'conditional_mean_confidence':float(np.mean(wt[:,central][wt[:,central]>0])),'static_support_frames':support.tolist()}
 stats={key:{'max_absolute_gain_db':float(np.max(abs(g[:,central]))),'mean_absolute_gain_db':float(np.mean(abs(g[:,central]))),'rate_p95_db_per_s':float(np.percentile(abs(np.diff(g[:,central],axis=1))/(H/FS),95)),'rate_max_db_per_s':float(np.max(abs(np.diff(g[:,central],axis=1))/(H/FS)))} for key,g in [('static',sg),('temporal',tg)]}
 rows.append({'device':d,'group':clip['group'],'label':clip['label'],'start_s':a,'anchor_db':anchor,'scores':scores,'stages':phases,'peak_contrast':peaks,'gain_stats':stats,'observation':observation,'media':med,'envelope_xcorr_lag_ms':lag,'common_safety_scale':float(scale),'preview_true_peak_dbfs':float(20*np.log10(maxtrue*scale))})
 grids.append({'device':d,'group':clip['group'],'times_clip_s':(tt[central]-.5).tolist(),'centers_hz':cs.tolist(),'static_gain_db':sg[:,central].tolist(),'temporal_gain_db':tg[:,central].tolist(),'confidence':wt[:,central].tolist()});qa.append({'device':d,'group':clip['group'],'baseline_reconstruction_max_abs':float(max(abs(recon[crop]-base[crop]))),'pre_scale_true_peak_dbfs':float(20*np.log10(maxtrue)),'preview_peak_dbfs':float(20*np.log10(maxsample*scale)),'preview_true_peak_dbfs':float(20*np.log10(maxtrue*scale))});print(d,clip['group'],{key:round(v['frame_mae_db'],3) for key,v in scores.items()},flush=True)
report={'scope':'Reference-assisted same-clip diagnostic fit; KM184 used at processing time. No deployed filter, input-only predictability, held-out performance, phase restoration or generalization claim. October only.','rules':(Path(__file__).with_name('REFERENCE_DIAGNOSTIC_PLAN.md')).read_text(),'centers_hz':cs.tolist(),'clips':rows,'gain_grids':grids,'qa':qa,'baseline_filter_hashes':{d:hashlib.sha256(np.array(prior['models'][d]['oct_12']['sos']).tobytes()).hexdigest() for d in ('iphone','s21')}}
(OUT/'reference-magnitude-diagnostic.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
# Compact UI data omits full matrices, still offers reproducible JSON.
ui={key:v for key,v in report.items() if key not in ('gain_grids','rules')};(OUT/'reference-magnitude-browser.json').write_text(json.dumps(ui,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n')
