"""Short attack-band balance check for the two displayed percussion hits.

Descriptive QC of the fixed 1792 s preview, not a training target or a
general percussion score. Ratios cancel common playback gain.
"""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/experiments/20261008-iphone-km184-public'
FS = 48000
VARIANTS = ('raw', 'baseline', 'broad_cut', 'shaped')
BANDS = ((800, 1600), (1600, 3200), (3200, 6000))

def read(device, variant):
    y, fs = sf.read(OUT / f'uppermid-percussion-1792-{device}-{variant}.wav')
    assert fs == FS and len(y) == 5 * FS
    return y

def filtered(y):
    return [signal.sosfilt(signal.butter(3, band, btype='bandpass', fs=FS, output='sos'), y)
            for band in BANDS]

ref = read('iphone', 'km184')
high = signal.sosfilt(signal.butter(3, 700, btype='highpass', fs=FS, output='sos'), ref)
frames = high.reshape(-1, 480)
level = 20 * np.log10(np.maximum(np.sqrt(np.mean(frames ** 2, axis=1)), 1e-8))
peaks, properties = signal.find_peaks(level, prominence=10, distance=20, height=np.median(level) + 8)
times = [round(float(p / 100), 3) for p in peaks]
assert len(times) == 2, times

result = {'scope': 'The two selected percussion hits within KM184 1792–1797s; October pair only',
          'method': 'KM184 >700Hz 10ms RMS local peaks; attack window peak−10ms to peak+50ms. Third-order Butterworth bandpass RMS at 800–1600, 1600–3200 and 3200–6000Hz. Device upper/lower dB ratio minus KM184 upper/lower dB ratio. Same two hit times and comparison scale for every variant.',
          'cautions': ['Two hits are too few to generalize to the whole percussion technique.',
                       'Peak-centered 60ms windows are not manually verified physical onsets.',
                       'This is a relative attack-band balance, not overall spectrum, loudness or proof of audible causation.'],
          'peak_times_within_clip_s': times, 'devices': {}}
ref_bands = filtered(ref)
for device in ('iphone', 's21'):
    variants = {}
    for variant in VARIANTS:
        xb = filtered(read(device, variant))
        events = []
        for t in times:
            lo = round((t - .01) * FS)
            hi = round((t + .05) * FS)
            rms = lambda z: float(np.sqrt(np.mean(z[lo:hi] ** 2)))
            delta = [20 * np.log10(max(rms(x), 1e-12) / max(rms(y), 1e-12)) for x, y in zip(xb, ref_bands)]
            events.append({'time_s': t, 'device_minus_km_db': [round(float(v), 3) for v in delta],
                           'upper_1600_3200_relative_to_800_1600_db': round(float(delta[1] - delta[0]), 3),
                           'upper_3200_6000_relative_to_800_1600_db': round(float(delta[2] - delta[0]), 3)})
        variants[variant] = {'events': events,
                             'median_1600_3200_relative_db': round(float(np.median([e['upper_1600_3200_relative_to_800_1600_db'] for e in events])), 3),
                             'median_3200_6000_relative_db': round(float(np.median([e['upper_3200_6000_relative_to_800_1600_db'] for e in events])), 3)}
    result['devices'][device] = variants
(OUT / 'percussion-attack-qc.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
for device, variants in result['devices'].items():
    print(device, {name: (v['median_1600_3200_relative_db'], v['median_3200_6000_relative_db'])
                   for name, v in variants.items()})
