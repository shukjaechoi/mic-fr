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
background=json.loads((LOCAL/'guitar-background-v2.json').read_text())
floor_psd={d:np.asarray(background['median_psd'][d],float) for d in ('km184','iphone')}
for d in ('s21','h1','d10'):
    estimates=[]
    for quiet in background['selected']:
        t=quiet['start_s'];m=maps[d]
        start=round(t*FS*m['slope']+m.get('interceptSamples',m.get('intercept_samples',0)))
        x=np.asarray(source[d][start:start+2*FS],float)
        estimates.append(signal.welch(x,FS,nperseg=8192,noverlap=4096)[1])
    floor_psd[d]=np.median(estimates,axis=0)
freq=np.fft.rfftfreq(8192,1/FS)
full_bands=((100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6000))
band_widths=np.asarray([hi-lo for lo,hi in full_bands],float)
band_slices=[(freq>=lo)&(freq<hi) for lo,hi in full_bands]
floor_bands={d:np.array([np.mean(f[b]) for b in band_slices]) for d,f in floor_psd.items()}
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
    f,p=signal.periodogram(y,FS,window='hann',nfft=8192,scaling='density')
    p=p+1e-20
    def energy(lo,hi): return float(np.mean(p[(f>=lo)&(f<hi)]))
    all_band_psd=[energy(lo,hi) for lo,hi in full_bands]
    e0,e1,e2=all_band_psd[3:]
    sel=(f>=800)&(f<6000)
    db=10*np.log10(p[sel])
    return {'band_psd':all_band_psd,
            'upper_16_32_over_08_16_db':round(10*np.log10(e1/e0),3),
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
                if v=='raw':
                    psd=np.asarray(metrics[stage]['band_psd'])
                    metrics[stage]['band_snr_db']=(10*np.log10(np.maximum(psd/floor_bands[d],1e-30))).round(3).tolist()
            records[d]['variants'][v]=metrics
    rows.append({'group':group,'km184_peak_time_s':t,'devices':records})

def confidence(km,dev,metric,fade_db=36):
    bins={'upper_16_32_over_08_16_db':(3,4),'upper_32_60_over_08_16_db':(3,5),
          'peak_background_contrast_db':(3,4,5)}[metric]
    snr=min(*(km['band_snr_db'][i] for i in bins),*(dev['band_snr_db'][i] for i in bins))
    kp=np.asarray(km['band_psd']);dp=np.asarray(dev['band_psd'])
    # A ratio or contrast involving a weak band receives less decision weight,
    # even if that band is measurable above the acoustic background.
    ke=kp*band_widths;de=dp*band_widths
    strength=min(max(10*np.log10(ke[i]/max(ke)),10*np.log10(de[i]/max(de))) for i in bins)
    importance=float(.2+.8*np.clip((strength+fade_db)/(fade_db-12),0,1))
    return round(float(np.clip((snr-10)/15,0,1)*importance),3),round(float(snr),2)

for row in rows:
    for d,record in row['devices'].items():
        for variant,by_stage in record['variants'].items():
            for stage,metrics in by_stage.items():
                km=row['devices']['km184']['variants']['raw'][stage]
                dev=record['variants']['raw'][stage]
                metrics['confidence']={k:{'weight':confidence(km,dev,k)[0],
                                          'joint_min_snr_db':confidence(km,dev,k)[1]} for k in
                                       ('upper_16_32_over_08_16_db','upper_32_60_over_08_16_db','peak_background_contrast_db')}

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
                stage_result[stage]={}
                for k,vals in diffs.items():
                    weights=np.array([r['devices'][d]['variants'][variant][stage]['confidence'][k]['weight'] for r in rr])
                    good=weights>0
                    stage_result[stage][k]={'median_db':round(float(np.median(vals)),3),
                        'confidence_weighted_mean_db':round(float(np.average(np.asarray(vals)[good],weights=weights[good])),3) if np.any(good) else None,
                        'valid_events':int(good.sum()),'mean_confidence':round(float(np.mean(weights[good])),3) if np.any(good) else 0,
                        'min_db':round(float(min(vals)),3),'max_db':round(float(max(vals)),3),
                        'positive_events':int(sum(v>0 for v in vals)),'n':len(vals)}
            summary[group][d][variant]=stage_result

payload={'scope':'October 2, matched guitar events in four labeled groups, 10 events from one session. Descriptive hypothesis check only.',
         'time_alignment':'Affine sample clock map followed by each device local >700Hz 10ms RMS peak within ±70ms; local peak is not certified physical onset.',
         'stages_ms':{'attack':[0,40],'body':[40,150],'tail':[150,300]},
         'metrics':'Band power ratios are dB and invariant to overall gain. Peak/background contrast is 90th minus 50th percentile of log FFT power over 800–6000Hz; it is spectral concentration, not certified harmonicity. Tail may include the next arpeggio note.',
         'confidence':'For each event-stage, both KM184 and device raw band PSD must exceed their October background proxy by 10dB in every band used by the metric. Above that, observability weight rises linearly from 0 to 1 across 10–25dB minimum joint SNR. Relevance compares band total energy (mean PSD × bandwidth) against the strongest of six non-overlapping 100–6000Hz bands in at least one recorder, not merely the upper bands: weight is 1 within 12dB and fades to 0.2 by 36dB below the strongest. Thus weaker but clearly observed upper components remain visible with reduced decision weight. The product weights the comparison. Same raw-derived weight is used for pre/post EQ. Weighted means exclude zero-weight events; null means insufficient coverage. Background is a recorded acoustic proxy, not microphone self-noise.',
         'cautions':['Pitched and percussive events differ; do not call percussive peaks harmonics.','Source/microphone distance, direction, room and phone processing differ.','An EQ-induced ratio change does not demonstrate perceptual improvement.','Sparse events, especially arpeggio/percussion, do not establish generality.'],
         'events':rows,'summary':summary}
sensitivity={}
for group,d,stage,key in (('arpeggio','iphone','attack','upper_16_32_over_08_16_db'),
                          ('complex','iphone','body','upper_16_32_over_08_16_db'),
                          ('arpeggio','s21','attack','upper_32_60_over_08_16_db')):
    rr=[r for r in rows if r['group']==group]
    case=f'{group}/{d}/{stage}/{key}'
    sensitivity[case]={}
    for taper in (24,36,48):
        weights=[];values={'raw':[],'fixed_eq':[]}
        for r in rr:
            km=r['devices']['km184']['variants']['raw'][stage]
            raw=r['devices'][d]['variants']['raw'][stage]
            weights.append(confidence(km,raw,key,taper)[0])
            for variant in values:
                values[variant].append(r['devices'][d]['variants'][variant][stage][key]-km[key])
        good=np.asarray(weights)>0
        sensitivity[case][str(taper)]={variant:round(float(np.average(np.asarray(vals)[good],weights=np.asarray(weights)[good])),3)
                                       if np.any(good) else None for variant,vals in values.items()}
payload['relevance_taper_sensitivity_db']=sensitivity
(OUT/'joint-event-timbre-qc.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
for g in summary:
    print(g,len([r for r in rows if r['group']==g]))
    for d in ('iphone','s21'):
        for stage in ('attack','body','tail'):
            x=summary[g][d]['raw'][stage]
            print(d,stage,*(x[k]['median_db'] for k in keys))
