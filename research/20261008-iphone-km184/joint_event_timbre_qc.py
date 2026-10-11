"""Joint spectral/time diagnostics on corresponding October guitar events.

This describes matched recorded events. It does not identify the cause of a
perceived quality or recommend a particular processor from one feature alone.
"""
from pathlib import Path
import json
import numpy as np
from scipy import signal, ndimage

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/experiments/20261008-iphone-km184-public'
LOCAL=ROOT/'docs/experiments/20261008-iphone-km184'
FS=48000
old=json.loads((ROOT/'docs/experiments/20261002/data/analysis.json').read_text())
maps={m['id']:m['guitarSync'] for m in old['microphones']}
maps['iphone']=json.loads((LOCAL/'analysis.json').read_text())['sets']['guitar']['mapping']
profiles=json.loads((OUT/'upper-mid-supplement-browser.json').read_text())['models']
source={d:np.load(f'/private/tmp/mic-audit-20261008/{d}-guitar.npy',mmap_mode='r') for d in ('km184','iphone','s21','h1','d10')}
events=[('open_notes',242.315),('open_notes',363.485),
        ('arpeggio',102.425),('arpeggio',102.875),('arpeggio',103.450),
        ('complex',1704.460),('complex',1706.445),('complex',1706.840),
        ('percussion',1792.430),('percussion',1794.915)]
stages=[('attack',0,.04),('body',.04,.15),('tail',.15,.30)]

def get_event(d,t):
    m=maps[d]; mapped=(t*FS*m['slope']+m.get('interceptSamples',m.get('intercept_samples',0)))/FS
    start=round((mapped-.12)*FS)
    x=np.asarray(source[d][start:start+round(.55*FS)],float)
    assert len(x)==round(.55*FS)
    hp=signal.sosfiltfilt(signal.butter(3,700,btype='highpass',fs=FS,output='sos'),x)
    env=np.sqrt(ndimage.uniform_filter1d(hp*hp,size=480))
    center=round(.12*FS);radius=round(.07*FS)
    peak=np.argmax(env[center-radius:center+radius])+center-radius
    return x,peak,round((peak-center)/FS*1000,2)

def spectrum_metrics(y):
    # All metrics are level-invariant within the stage, so display gain cancels.
    n=len(y);nfft=8192
    p=np.abs(np.fft.rfft(y*np.hanning(n),n=nfft))**2+1e-20
    f=np.fft.rfftfreq(nfft,1/FS)
    def energy(lo,hi): return float(np.mean(p[(f>=lo)&(f<hi)]))
    e0,e1,e2=energy(800,1600),energy(1600,3200),energy(3200,6000)
    sel=(f>=800)&(f<6000)
    db=10*np.log10(p[sel])
    return {'upper_16_32_over_08_16_db':round(10*np.log10(e1/e0),3),
            'upper_32_60_over_08_16_db':round(10*np.log10(e2/e0),3),
            'peak_background_contrast_db':round(float(np.percentile(db,90)-np.percentile(db,50)),3)}

rows=[]
for group,t in events:
    records={}
    for d in source:
        x,peak,offset=get_event(d,t)
        variants={'raw':x}
        if d in ('iphone','s21'):
            variants['fixed_eq']=signal.sosfilt(np.asarray(profiles[d]['sos']['shaped'],float),x)
        records[d]={'local_peak_offset_ms':offset,'variants':{}}
        for v,y in variants.items():
            metrics={}
            for stage,a,b in stages:
                aa=peak+round(a*FS);bb=peak+round(b*FS)
                metrics[stage]=spectrum_metrics(y[aa:bb])
            records[d]['variants'][v]=metrics
    rows.append({'group':group,'km184_peak_time_s':t,'devices':records})

keys=('upper_16_32_over_08_16_db','upper_32_60_over_08_16_db','peak_background_contrast_db')
summary={}
for group in sorted({r['group'] for r in rows}):
    summary[group]={}
    rr=[r for r in rows if r['group']==group]
    for d in ('iphone','s21','h1','d10'):
        summary[group][d]={}
        for variant in ('raw','fixed_eq'):
            if variant=='fixed_eq' and d not in ('iphone','s21'): continue
            stage_result={}
            for stage,_,_ in stages:
                diffs={k:[r['devices'][d]['variants'][variant][stage][k]-r['devices']['km184']['variants']['raw'][stage][k] for r in rr] for k in keys}
                stage_result[stage]={k:{'median_db':round(float(np.median(vals)),3),'min_db':round(float(min(vals)),3),'max_db':round(float(max(vals)),3),'positive_events':int(sum(v>0 for v in vals)),'n':len(vals)} for k,vals in diffs.items()}
            summary[group][d][variant]=stage_result

payload={'scope':'October 2, matched guitar events in four labeled groups, 10 events from one session. Descriptive hypothesis check only.',
         'time_alignment':'Affine sample clock map followed by each device local >700Hz 10ms RMS peak within ±70ms; local peak is not certified physical onset.',
         'stages_ms':{'attack':[0,40],'body':[40,150],'tail':[150,300]},
         'metrics':'Band power ratios are dB and invariant to overall gain. Peak/background contrast is 90th minus 50th percentile of log FFT power over 800–6000Hz; it is spectral concentration, not certified harmonicity. Tail may include the next arpeggio note.',
         'cautions':['Pitched and percussive events differ; do not call percussive peaks harmonics.','Source/microphone distance, direction, room and phone processing differ.','An EQ-induced ratio change does not demonstrate perceptual improvement.','Sparse events, especially arpeggio/percussion, do not establish generality.'],
         'events':rows,'summary':summary}
(OUT/'joint-event-timbre-qc.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
for g in summary:
    print(g,len([r for r in rows if r['group']==g]))
    for d in ('iphone','s21'):
        for stage in ('attack','body','tail'):
            x=summary[g][d]['raw'][stage]
            print(d,stage,*(x[k]['median_db'] for k in keys))
