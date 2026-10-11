"""Descriptive percussion decay QC; aligns each device at its local hit peak."""
from collections import defaultdict
from pathlib import Path
import json, subprocess, sys
import numpy as np
import soundfile as sf
from scipy import signal, ndimage

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'docs/experiments/20261008-iphone-km184-public'
SRC=Path('/Users/sjchoi/dev/rec/261002-iphone,h1,km184,s21,d10')
FS=48000
HITS=[1772.92,1774.46,1776.795,1779.645,1782.65,1785.135,1788.035,1790.445,1792.43,1794.915,1797.155,1799.28,1801.325]
BANDS=[(200,800),(800,1600),(1600,3200),(3200,6000)]
FILES={'km184':'neumann km184 guitar.wav','h1':'h1 guitar.WAV','d10':'sony pcm d10 guitar.wav','s21':'s21 guitar.wav'}
old=json.loads((ROOT/'docs/experiments/20261002/data/analysis.json').read_text())
maps={m['id']:m['guitarSync'] for m in old['microphones']}
maps['iphone']=json.loads((ROOT/'docs/experiments/20261008-iphone-km184/analysis.json').read_text())['sets']['guitar']['mapping']
tail=Path('/private/tmp/iphone-percussion-40s-stereo.wav')
if not tail.exists():
    sys.path.insert(0,'/private/tmp/mic-analysis-deps')
    import imageio_ffmpeg
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-ss','1712','-i',str(SRC/'iphone guitar.m4a'),'-t','40','-c:a','pcm_f32le',str(tail)],check=True)
data={}
for d in ('km184','iphone','s21','h1','d10'):
    rows=[]
    for hit in HITS:
        m=maps[d]
        ts=(hit*FS*m['slope']+m.get('interceptSamples',m.get('intercept_samples',0)))/FS
        path=tail if d=='iphone' else SRC/FILES[d]
        origin=1712 if d=='iphone' else 0
        with sf.SoundFile(path) as f:
            f.seek(round((ts-origin-.11)*FS))
            stereo=f.read(round(.35*FS),always_2d=True)
        mono=stereo.mean(axis=1)
        corr=float(np.corrcoef(stereo[:,0],stereo[:,1])[0,1]) if stereo.shape[1]==2 else None
        attenuation=float(20*np.log10(np.sqrt(np.mean(mono**2))/np.sqrt(np.mean(stereo**2)))) if stereo.shape[1]==2 else None
        hp=signal.sosfiltfilt(signal.butter(3,700,btype='highpass',fs=FS,output='sos'),mono)
        env=np.sqrt(ndimage.uniform_filter1d(hp**2,size=480))
        center=round(.11*FS)
        peak=np.argmax(env[center-round(.06*FS):center+round(.06*FS)])+center-round(.06*FS)
        ratios=[];curves=[]
        for lo,hi in BANDS:
            z=signal.sosfiltfilt(signal.butter(3,[lo,hi],btype='bandpass',fs=FS,output='sos'),mono)
            early=np.mean(z[peak:peak+round(.03*FS)]**2)
            late=np.mean(z[peak+round(.03*FS):peak+round(.18*FS)]**2)
            ratios.append(float(10*np.log10((late+1e-12)/(early+1e-12))))
            power=ndimage.uniform_filter1d(z**2,size=240)
            samples=peak+np.arange(37)*240
            curves.append((10*np.log10((power[samples]+1e-12)/(early+1e-12))).tolist())
        rows.append({'km184_time_s':hit,'local_peak_offset_ms':round((peak-center)/FS*1000,2),'stereo_correlation':corr,'mono_attenuation_db':attenuation,'late_30_180_over_early_0_30_db':ratios,'curve_db':curves})
    data[d]=rows
ref=np.array([r['late_30_180_over_early_0_30_db'] for r in data['km184']])
summary={}
for d,rows in data.items():
    arr=np.array([r['late_30_180_over_early_0_30_db'] for r in rows]);delta=arr-ref
    curves=np.array([r['curve_db'] for r in rows])
    corrs=[r['stereo_correlation'] for r in rows if r['stereo_correlation'] is not None]
    atts=[r['mono_attenuation_db'] for r in rows if r['mono_attenuation_db'] is not None]
    summary[d]={'late_early_median_db':np.median(arr,axis=0).round(3).tolist(),
                'vs_km184_median_db':np.median(delta,axis=0).round(3).tolist(),
                'more_late_energy_events':(delta>0).sum(axis=0).tolist(),
                'median_curve_db':np.median(curves,axis=0).round(3).tolist(),
                'stereo_corr_median':round(float(np.median(corrs)),3) if corrs else None,
                'mono_rms_attenuation_median_db':round(float(np.median(atts)),3) if atts else None}
payload={'scope':'October 2 percussion, 13 prominent KM184 highpass peaks in 1772-1802 s. Descriptive comparison, not a causal attribution or generalization result.',
         'alignment':'Affine clock map followed by separate local >700Hz 10ms RMS maximum within ±60ms of each mapped KM184 peak; ratios computed from local peak. Local peak may not be acoustic onset.',
         'metric':'Band-limited mean-square energy: 30-180ms divided by 0-30ms from local peak; dB. Positive device minus KM184 means proportionally more late energy. Different source/mic placement and postprocessing can contribute.',
         'bands_hz':BANDS,'curve_time_ms':list(range(0,181,5)),'events':data,'summary':summary}
(OUT/'temporal-spread-qc.json').write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
print({d:v['vs_km184_median_db'] for d,v in summary.items()})
