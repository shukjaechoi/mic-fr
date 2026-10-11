"""Frozen October EQ plus causal 5 ms attack controllers; percussion excluded from fit."""
from pathlib import Path
import hashlib, json
import numpy as np
import soundfile as sf
from scipy import signal, ndimage
import torch

ROOT=Path(__file__).resolve().parents[2]
PUB=ROOT/'docs/experiments/20261008-iphone-km184-public'
LOC=ROOT/'docs/experiments/20261008-iphone-km184'
FS=48000; BLOCK=240
base=json.loads((PUB/'upper-mid-supplement-browser.json').read_text())
all_scores=json.loads((LOC/'october-four-device-eq.json').read_text())['scores']
old=json.loads((ROOT/'docs/experiments/20261002/data/analysis.json').read_text())
maps={m['id']:m['guitarSync'] for m in old['microphones']}
maps['iphone']=json.loads((LOC/'analysis.json').read_text())['sets']['guitar']['mapping']
ref=np.load('/private/tmp/mic-audit-20261008/km184-guitar.npy',mmap_mode='r')
sources={d:np.load(f'/private/tmp/mic-audit-20261008/{d}-guitar.npy',mmap_mode='r') for d in ('iphone','s21')}
times={'open_notes':418,'complex':1704,'percussion':1792}
bands={'iphone':(1600,3200),'s21':(3200,6000)}

def window(device,t):
    ix=round(t*FS); m=maps[device]
    ps=round(ix*m['slope']+m.get('intercept_samples',m.get('interceptSamples',0)))
    left=max(0,ps-12000); off=ps-left
    x=np.asarray(sources[device][left:ps+5*FS],dtype=np.float64)
    sos=np.asarray(base['models'][device]['sos']['shaped'])
    z=signal.sosfilt(sos,x)[off:]
    r=np.asarray(ref[ix:ix+5*FS],dtype=np.float64)
    if len(z)!=5*FS or len(r)!=5*FS: raise ValueError((device,t))
    return z,r

def powers(x,device):
    lo=signal.sosfilt(signal.butter(3,[800,1600],btype='bandpass',fs=FS,output='sos'),x)
    hi=signal.sosfilt(signal.butter(3,bands[device],btype='bandpass',fs=FS,output='sos'),x)
    def rms(y): return np.sqrt(np.mean(y.reshape(-1,BLOCK)**2,axis=1)+1e-12)
    return 20*np.log10(rms(x)),20*np.log10(rms(lo)),20*np.log10(rms(hi))

def features(z,device):
    level,low,high=powers(z,device)
    lag=ndimage.uniform_filter1d(level,size=5,mode='nearest')
    prior=np.r_[np.repeat(level[0],4),level[:-4]]
    novelty=level-prior
    hprior=np.r_[np.repeat(high[0],4),high[:-4]]
    X=np.stack([level,low,high,high-low,novelty,high-hprior],axis=1)
    # 5ms decision blocks, one-block processing latency. Gate uses device signal only.
    trigger=(novelty>6)&(level>-42)&(high>-62)
    gate=ndimage.maximum_filter1d(trigger.astype(int),size=10,origin=4,mode='constant').astype(bool)
    return X,gate

def apply(z,device,db):
    band=signal.sosfilt(signal.butter(3,bands[device],btype='bandpass',fs=FS,output='sos'),z)
    # First-order dezipper; negligible spectral change outside selected band.
    g=10**(np.clip(db,-2,0)/20)
    g=np.repeat(g,BLOCK)
    g=signal.lfilter([1/120],[1,-119/120],g-1)+1
    return z+(g-1)*band

def ratio(y,r,device):
    _,l,h=powers(y,device); _,rl,rh=powers(r,device)
    return (h-l)-(rh-rl)

result={'method':'Frozen shaped SOS + 5ms-block upper-band cut; one-block buffering (5ms) for current-block features and causal 50ms hold; max 2dB cut. Training excludes percussion and complex chord; existing SOS was previously exposed to October data, so this is a controlled development check, not independent validation.', 'bands_hz':bands,'models':{},'clips':[]}
torch.set_num_threads(2)
for d in ('iphone','s21'):
    groups={}
    for row in all_scores[d]:
        g=row['group']; t=row['start_s']
        if g in ('percussion','complex','piece'): continue
        groups.setdefault(g,[]).append(t)
    # Reduce repeated neighboring windows, preserving diverse techniques.
    feats=[];targets=[];weights=[]
    for g,ts in groups.items():
        this=[]
        for t in ts[::max(1,len(ts)//12)]:
            z,r=window(d,t); X,gate=features(z,d)
            rr=ratio(z,r,d)
            keep=gate & np.isfinite(rr)
            if np.any(keep): this.append((X[keep],np.clip(-rr[keep],-2,0)))
        if this:
            xx=np.concatenate([v[0] for v in this]); yy=np.concatenate([v[1] for v in this]);
            feats.append(xx);targets.append(yy);weights.append(np.full(len(yy),1/len(yy)))
    X=np.concatenate(feats);Y=np.concatenate(targets);W=np.concatenate(weights)
    mean=X.mean(axis=0);std=np.maximum(X.std(axis=0),1)
    xx=torch.tensor((X-mean)/std,dtype=torch.float32)
    yy=torch.tensor(Y,dtype=torch.float32); ww=torch.tensor(W/W.mean(),dtype=torch.float32)
    torch.manual_seed(42)
    net=torch.nn.Sequential(torch.nn.Linear(6,12),torch.nn.Tanh(),torch.nn.Linear(12,1),torch.nn.Sigmoid())
    opt=torch.optim.Adam(net.parameters(),lr=.006)
    for _ in range(140):
        pred=-2*net(xx).squeeze()
        loss=(ww*torch.nn.functional.smooth_l1_loss(pred,yy,reduction='none')).mean()+.025*(pred**2).mean()
        opt.zero_grad();loss.backward();opt.step()
    weights_json=[]
    for layer in (net[0],net[2]):
        weights_json.append({'weight':layer.weight.detach().numpy().tolist(),'bias':layer.bias.detach().numpy().tolist()})
    result['models'][d]={'band_hz':bands[d],'trained_groups':list(groups),'trained_attack_blocks':len(Y),'feature_mean':mean.tolist(),'feature_std':std.tolist(),'mlp':weights_json,'max_cut_db':-2,'simple_dynamic_cut_db':-2}
    for g,t in times.items():
        z,r=window(d,t);X0,gate=features(z,d)
        simple=np.where(gate,-2.,0.)
        with torch.no_grad(): pred=-2*net(torch.tensor((X0-mean)/std,dtype=torch.float32)).squeeze().numpy()
        neural=np.where(gate,pred,0.)
        ys={'km184':r,'shaped':z,'dynamic':apply(z,d,simple),'neural':apply(z,d,neural)}
        # Preserve earlier raw for convenience and use same display anchor/safety for all.
        oldclip=next(c for c in base['clips'] if c['group']==g and c['device']==d)
        raw,_=sf.read(PUB/oldclip['media']['raw']['audio'])
        ys['raw']=raw/oldclip['common_display_scale'] / (10**(oldclip['display_anchor_db']/20))
        f,rp=signal.welch(r,FS,nperseg=8192,noverlap=4096)
        _,xp=signal.welch(ys['raw'],FS,nperseg=8192,noverlap=4096)
        anchors=[]
        for c in (335.7,597.7):
            sel=(f>=c/2**(1/6))&(f<c*2**(1/6))
            anchors.append(10*np.log10(np.mean(rp[sel])/np.mean(xp[sel])))
        gain=10**(np.median(anchors)/20)
        for k in ys:
            if k!='km184':ys[k]=ys[k]*gain
        scale=min(1,.8/max(np.max(abs(v)) for v in ys.values()))
        for k in ys:ys[k]=ys[k]*scale
        media={}
        for k,v in ys.items():
            name=f'attack-{g}-{t}-{d}-{k}.wav'
            sf.write(PUB/name,v,FS,subtype='PCM_16')
            media[k]={'audio':name}
        # Both entire clip and two main attack times; descriptive metrics only.
        x={'group':g,'start_s':t,'device':d,'gated_blocks':int(gate.sum()),'neural_mean_cut_db_on_gate':float(np.mean(neural[gate])) if gate.any() else 0,'media':media}
        if g=='percussion':
            event=[]
            for center in (.42,2.91):
                a=max(0,int((center-.01)*FS));b=min(5*FS,int((center+.05)*FS));
                event.append({k:float(np.mean(ratio(v[a//BLOCK*BLOCK:b//BLOCK*BLOCK],ys['km184'][a//BLOCK*BLOCK:b//BLOCK*BLOCK],d))) for k,v in ys.items() if k!='km184'})
            x['attack_excess_db']=event
        result['clips'].append(x)
    print(d,'training blocks',len(Y),'percussion',result['clips'][-1].get('attack_excess_db'),flush=True)
(PUB/'attack-controller-trial.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
