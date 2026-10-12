"""October13-hit diagnostic: stage envelopes, observable resonances, input-only gain predictability."""
from pathlib import Path
import json
import numpy as np
from scipy import signal,ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
source=Path(__file__).with_name('reference_magnitude_diagnostic.py').read_text();exec(compile(source[:source.index('rows=[];grids=[];qa=[]')],str(Path(__file__).with_name('reference_magnitude_diagnostic.py')),'exec'))
events=json.loads((OUT/'percussion-response-eq.json').read_text())['events'];bands6=[(600,1200),(1200,2400),(2400,4800),(4800,8000),(8000,12000)];records=[];training={d:[] for d in ('iphone','s21')};curves={d:[] for d in ('km184','iphone','s21')};dt=.005;grid=np.arange(-.05,.301,dt)
for e in events:
 a=e['start_s'];duration=.35;ref=read('km184',a-.5,a+.4);crop=slice(24000,43200);raws={d:read(d,a-.5,a+.4) for d in ('iphone','s21')};base={d:read(d,a-.5,a+.4,sos=np.array(prior['models'][d]['oct_12']['sos'])) for d in raws}
 # Use the already fixed percussion clip anchor, avoiding eventwise gain normalization.
 anchors={d:next(c['anchor_db'] for c in prior['clips'] if c['device']==d and c['group']=='percussion') for d in raws}
 for d in base:base[d]*=10**(anchors[d]/20)
 for d,x in [('km184',ref),*base.items()]:
  ev=[]
  for lo,hi in bands6:
   ss=signal.butter(3,[lo,hi],btype='bandpass',fs=FS,output='sos');filtered=signal.sosfiltfilt(ss,x);eng=ndimage.uniform_filter1d(filtered**2,240);ev.append(np.interp(grid,np.arange(len(eng))/FS-.5,eng))
  curves[d].append(np.array(ev))
 for d in raws:
  kcurve=curves['km184'][-1];xcurve=curves[d][-1];stage={};valid=[]
  for bi,(lo,hi) in enumerate(bands6):
   mask=(ff>=lo)&(ff<hi);bk=np.sum(floors['km184'][mask])*(ff[1]-ff[0]);bx=np.sum(floors[d][mask])*(ff[1]-ff[0])*10**(anchors[d]/10);stage[str(lo)+'_'+str(hi)]={}
   for name,gate in [('attack',(grid>=0)&(grid<.04)),('body',(grid>=.04)&(grid<.15)),('decay',(grid>=.15)&(grid<=.3))]:
    ke=float(np.mean(kcurve[bi,gate]));xe=float(np.mean(xcurve[bi,gate]));sn=min(10*np.log10(max(ke/bk,1e-30)),10*np.log10(max(xe/bx,1e-30)));stage[str(lo)+'_'+str(hi)][name]={'signed_device_minus_km_db':float(10*np.log10(max(xe/ke,1e-30))) if sn>=10 else None,'joint_snr_db':float(sn)}
   v=(grid>=0)&(grid<.15);kt=float(np.sum(grid[v]*kcurve[bi,v])/np.sum(kcurve[bi,v]));xt=float(np.sum(grid[v]*xcurve[bi,v])/np.sum(xcurve[bi,v]));valid.append({'band_hz':[lo,hi],'energy_centroid_difference_ms':(xt-kt)*1000})
  # Observable narrow spectral features over300ms; not short-attack harmonic inference.
  kp=signal.welch(ref[24000:38400],FS,nperseg=8192,noverlap=4096)[1];xp=signal.welch(base[d][24000:38400],FS,nperseg=8192,noverlap=4096)[1];rp=signal.welch(raws[d][24000:38400],FS,nperseg=8192,noverlap=4096)[1];db=10*np.log10(np.maximum(kp,1e-30));peaks,_=signal.find_peaks(db,prominence=6,distance=8);features=[]
  for pi in peaks:
   hz=ff[pi];core=abs(ff-hz)<=25;flank=(abs(ff-hz)>=50)&(abs(ff-hz)<=120)
   if not 700<hz<6000:continue
   sn=min([10*np.log10(np.mean(kp[m])/np.mean(floors['km184'][m])) for m in (core,flank)]+[10*np.log10(np.mean(rp[m])/np.mean(floors[d][m])) for m in (core,flank)])
   if sn<10:continue
   kc=10*np.log10(np.mean(kp[core])/np.mean(kp[flank]));xc=10*np.log10(np.mean(xp[core])/np.mean(xp[flank]));features.append({'frequency_hz':float(hz),'contrast_difference_db':float(xc-kc),'joint_snr_db':float(sn)})
  records.append({'id':e['id'],'device':d,'start_s':a,'split':e['split'],'stage_residuals':stage,'energy_centroids':valid,'narrow_features':features})
  # Frozen bounded oracle targets use identical1/6 confidence/limits as previous diagnostic.
  _,tt,zk=stft(ref);_,_,zr=stft(raws[d]);_,_,zb=stft(base[d]);pk,pr,pb=powers(zk),powers(zr),powers(zb);wt,snr=weights(pk,pr,d);wt*=wt>=.15;err=10*np.log10(np.maximum(bp(pk),1e-30)/np.maximum(bp(pb),1e-30));tg=smooth_obs(err,wt);num=ndimage.gaussian_filter1d(tg*wt,.06/(H/FS),axis=1);den=ndimage.gaussian_filter1d(wt,.06/(H/FS),axis=1);tg=np.clip(num/np.maximum(den,1e-12),-6,6)*np.clip(den/.5,0,1)
  for i in range(1,len(tt)):tg[:,i]=np.clip(tg[:,i],tg[:,i-1]-24*H/FS,tg[:,i-1]+24*H/FS)
  for i in range(len(tt)-2,-1,-1):tg[:,i]=np.clip(tg[:,i],tg[:,i+1]-24*H/FS,tg[:,i+1]+24*H/FS)
  # Device-only magnitude features; centered STFT has42.7ms lookahead (offline test).
  energy=np.maximum(bp(pr),1e-30);level=10*np.log10(np.sum(energy,axis=0));ratios=10*np.log10(energy)-level;flux=np.maximum(np.diff(np.log(energy),axis=1,prepend=np.log(energy[:,:1])),0);features=np.vstack([level,ratios,flux,np.roll(level,1),np.roll(ratios,1,axis=1)]).T;take=(tt>=.5)&(tt<=.8);training[d].append({'id':e['id'],'split':e['split'],'x':features[take],'target':tg[:,take].T,'weight':wt[:,take].T})
# One preselected ridge penalty, train8/eval5, no test-tuned features or penalty.
pred={};alpha=10
for d,rows in training.items():
 tr=[r for r in rows if r['split']=='design'];te=[r for r in rows if r['split']=='held_hit'];X=np.vstack([r['x'] for r in tr]);Y=np.vstack([r['target'] for r in tr]);W=np.vstack([r['weight'] for r in tr]);mu=X.mean(axis=0);sd=np.maximum(X.std(axis=0),.001);A=np.c_[np.ones(len(X)),(X-mu)/sd];coefs=[];constants=[]
 for j in range(len(cs)):
  w=W[:,j];pen=np.eye(A.shape[1])*alpha;pen[0,0]=0;coefs.append(np.linalg.solve(A.T@(A*w[:,None])+pen+1e-8*np.eye(A.shape[1]),A.T@(Y[:,j]*w)));constants.append(float(np.sum(Y[:,j]*w)/max(np.sum(w),1e-9)))
 coef=np.array(coefs).T;per=[]
 for r in te:
  p=np.clip(np.c_[np.ones(len(r['x'])),(r['x']-mu)/sd]@coef,-6,6);w=r['weight'];target=r['target'];per.append({'id':r['id'],'input_ridge_gain_mae_db':weighted(abs(p-target),w),'constant_gain_mae_db':weighted(abs(np.array(constants)-target),w),'zero_supplement_gain_mae_db':weighted(abs(target),w)})
 pred[d]={'scope':'Input-only prediction of reference-assisted gain, not output audio quality. First8 train/final5 developmental evaluation already explored. Centered STFT42.7ms lookahead; no real-time claim.','ridge_alpha':alpha,'feature_mean':mu.tolist(),'feature_scale':sd.tolist(),'ridge_coefficients':coef.tolist(),'constant_gain_db':constants,'centers_hz':cs.tolist(),'held_hits':per,'mean_gain_mae_db':{key:float(np.mean([r[key] for r in per if r[key] is not None])) for key in ('input_ridge_gain_mae_db','constant_gain_mae_db','zero_supplement_gain_mae_db')}}
# Summaries retain support and sign counts; unknown observations excluded, not zero.
summary={}
for d in ('iphone','s21'):
 rr=[r for r in records if r['device']==d];stages={}
 for lo,hi in bands6:
  key=str(lo)+'_'+str(hi);stages[key]={}
  for st in ('attack','body','decay'):
   vv=[r['stage_residuals'][key][st]['signed_device_minus_km_db'] for r in rr];vv=np.array([v for v in vv if v is not None]);stages[key][st]={'median_db':float(np.median(vv)) if len(vv) else None,'q25_db':float(np.percentile(vv,25)) if len(vv) else None,'q75_db':float(np.percentile(vv,75)) if len(vv) else None,'observed_hits':len(vv),'positive_hits':int(np.sum(vv>0)),'negative_hits':int(np.sum(vv<0))}
 # Broad fixed frequency buckets diagnose repeatable spectral contrast, not tunableEQ targets.
 narrow=[]
 for lo,hi in [(700,1000),(1000,1500),(1500,2000),(2000,3000),(3000,4500),(4500,6000)]:
  hitmed=[float(np.median([p['contrast_difference_db'] for p in r['narrow_features'] if lo<=p['frequency_hz']<hi])) for r in rr if any(lo<=p['frequency_hz']<hi for p in r['narrow_features'])];narrow.append({'band_hz':[lo,hi],'median_contrast_difference_db':float(np.median(hitmed)) if hitmed else None,'observed_hits':len(hitmed),'negative_hits':sum(v<0 for v in hitmed),'positive_hits':sum(v>0 for v in hitmed)})
 shift_qc={}
 for shift in (-.01,-.005,0,.005,.01):
  vals=[]
  for k,x in zip(curves['km184'],curves[d]):
   xx=np.array([np.interp(grid+shift,grid,row) for row in x]);gate=(grid>=.04)&(grid<.15);vals.append(10*np.log10(np.mean(xx[1,gate])/np.mean(k[1,gate])))
  shift_qc[str(round(shift*1000))]={'median_body_1200_2400_db':float(np.median(vals)),'positive_hits':int(np.sum(np.array(vals)>0)),'hits':len(vals)}
 summary[d]={'stages':stages,'narrow_contrast':narrow,'alignment_sensitivity_ms':shift_qc}
fig,axes=plt.subplots(5,2,figsize=(12,12),sharex=True)
for j,(lo,hi) in enumerate(bands6):
 k=np.median(np.array(curves['km184'])[:,j,:],axis=0);norm=max(k)
 for col,d in enumerate(('iphone','s21')):
  ax=axes[j,col];arr=np.array(curves[d])[:,j,:];med=np.median(arr,axis=0);ax.plot(grid*1000,10*np.log10(np.maximum(k/norm,1e-8)),label='KM184',color='#273f45');ax.plot(grid*1000,10*np.log10(np.maximum(med/norm,1e-8)),label=d,color='#148b86');ax.axvline(0,color='gray',ls=':');ax.axvline(40,color='gray',ls=':');ax.axvline(150,color='gray',ls=':');ax.set_ylim(-50,12);ax.set_title(f'{d} {lo}–{hi}Hz');ax.grid(alpha=.2);ax.legend(fontsize=8)
axes[-1,0].set_xlabel('Time from common onset (ms)');axes[-1,1].set_xlabel('Time from common onset (ms)');fig.supylabel('dB relative to KM184 median envelope peak (same per-band normalization)');fig.tight_layout();fig.savefig(OUT/'percussion-device-envelopes.png',dpi=140);plt.close(fig)
report={'scope':'13 October percussion hits, fixed latest1/12EQ and prior clip anchor. Stage/narrow diagnostics, no new corrective filter, no phase/reverb causal claim.','rules':'Stage0–40/40–150/150–300ms joint rawSNR>=10dB. Zero-phase bandpass+5ms energy smoothing descriptive only. Narrow core±25/flanks50–120Hz over300ms with8192 Welch, both core/flanksjointSNR>=10. Envelope median plotted with same reference normalization, not per-device level fitting. Input-only ridge alpha10 predefined, first8train/last5evaluation; centeredSTFT lookahead, repeated-session exploratory data. Gain prediction MAE is not waveform/timbre score.','summary':summary,'gain_prediction':pred,'events':records}
(OUT/'percussion-device-diagnosis.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n');print(json.dumps({'summary':summary,'prediction':pred},ensure_ascii=False,indent=2),flush=True)
