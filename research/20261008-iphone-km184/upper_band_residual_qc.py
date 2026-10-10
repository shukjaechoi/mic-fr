"""Diagnostic upper-band residuals of the frozen October four-device filters.

Uses the existing valid five-second window manifest. This is descriptive, not
an independent test or a new filter-design objective.
"""
from pathlib import Path
import json
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/experiments/20261008-iphone-km184'
FS = 48000
N = 8192
DEVICES = ('iphone', 's21', 'd10', 'h1')
report = json.loads((OUT / 'october-four-device-eq.json').read_text())
old = json.loads((ROOT / 'docs/experiments/20261002/data/analysis.json').read_text())
maps = {m['id']: m['guitarSync'] for m in old['microphones']}
maps['iphone'] = json.loads((OUT / 'analysis.json').read_text())['sets']['guitar']['mapping']
km = np.load('/private/tmp/mic-audit-20261008/km184-guitar.npy', mmap_mode='r')
audio = {k: np.load(f'/private/tmp/mic-audit-20261008/{k}-guitar.npy', mmap_mode='r') for k in DEVICES}
profiles = {k: np.asarray(report['models'][k]['sos'], float) for k in DEVICES if k != 'iphone'}
profiles['iphone'] = np.asarray(json.loads((OUT / 'october-final-filter.json').read_text())['sos'], float)
centers = 800 * 2 ** (np.arange(15) / 6)
freq = np.fft.rfftfreq(N, 1 / FS)
bands = [(freq >= c / 2 ** (1 / 12)) & (freq < c * 2 ** (1 / 12)) for c in centers]
anchor_centers = (335.7, 597.7)
anchor_bands = [(freq >= c / 2 ** (1 / 6)) & (freq < c * 2 ** (1 / 6)) for c in anchor_centers]

def power(x):
    return signal.welch(np.asarray(x, float), FS, nperseg=N, noverlap=N // 2)[1]

def band_power(p, bins):
    return np.array([np.mean(p[b]) for b in bins])

def valid(p, a):
    return 10 * np.log10(np.maximum(p / np.max(a), 1e-30)) >= -30

rows = []
for device in DEVICES:
    source = audio[device]
    sync = maps[device]
    for w in report['scores'][device]:
        t = w['start_s']
        ks = round(t * FS)
        ps = round(ks * sync['slope'] + sync.get('intercept_samples', sync.get('interceptSamples', 0)))
        pad = round(.25 * FS)
        x = np.asarray(source[max(0, ps - pad):ps + 5 * FS], float)
        off = ps - max(0, ps - pad)
        if len(x) - off != 5 * FS:
            continue
        ref = np.asarray(km[ks:ks + 5 * FS], float)
        raw = x[off:]
        corrected = signal.sosfilt(profiles[device], x)[off:]
        rp, xp, yp = map(power, (ref, raw, corrected))
        rb, xb, yb = [band_power(p, bands) for p in (rp, xp, yp)]
        ra, xa = band_power(rp, anchor_bands), band_power(xp, anchor_bands)
        mask = valid(rb, ra) & valid(xb, xa)
        anchor = float(np.median(10 * np.log10(np.maximum(ra / xa, 1e-30))))
        before = 10 * np.log10(np.maximum(rb / xb, 1e-30)) - anchor
        after = 10 * np.log10(np.maximum(rb / yb, 1e-30)) - anchor
        for j, c in enumerate(centers):
            if mask[j]:
                rows.append({'device': device, 'group': w['group'], 'start_s': t,
                             'center_hz': round(float(c), 1),
                             'before_km_minus_device_db': round(float(before[j]), 3),
                             'after_km_minus_device_db': round(float(after[j]), 3)})

summary = {}
for device in DEVICES:
    by_device = [r for r in rows if r['device'] == device]
    by_group = {}
    for group in sorted({r['group'] for r in by_device}):
        rr = [r for r in by_device if r['group'] == group]
        windows = len({r['start_s'] for r in rr})
        if not rr:
            continue
        before = np.array([r['before_km_minus_device_db'] for r in rr])
        after = np.array([r['after_km_minus_device_db'] for r in rr])
        by_group[group] = {'windows': windows, 'observed_window_bands': len(rr),
                           'before_mae_db': round(float(np.mean(abs(before))), 3),
                           'after_mae_db': round(float(np.mean(abs(after))), 3),
                           'after_signed_median_db': round(float(np.median(after)), 3)}
    by_band = []
    for c in centers:
        rr = [r for r in by_device if r['center_hz'] == round(float(c), 1)]
        if not rr:
            continue
        bef = np.array([r['before_km_minus_device_db'] for r in rr])
        aft = np.array([r['after_km_minus_device_db'] for r in rr])
        by_band.append({'center_hz': round(float(c), 1), 'windows': len(rr),
                        'before_median_db': round(float(np.median(bef)), 3),
                        'after_median_db': round(float(np.median(aft)), 3),
                        'after_mae_db': round(float(np.mean(abs(aft))), 3),
                        'after_positive_fraction': round(float(np.mean(aft > 0)), 3),
                        'after_q10_db': round(float(np.quantile(aft, .1)), 3),
                        'after_q90_db': round(float(np.quantile(aft, .9)), 3)})
    summary[device] = {'windows': len({r['start_s'] for r in by_device}),
                       'group_equal_before_mae_db': round(float(np.mean([v['before_mae_db'] for v in by_group.values()])), 3),
                       'group_equal_after_mae_db': round(float(np.mean([v['after_mae_db'] for v in by_group.values()])), 3),
                       'groups': by_group, 'bands': by_band}

result = {'scope': 'October matched guitar windows only; existing frozen full-fit filters; 800–4032 Hz',
          'method': '1/6-octave mean PSD, 5s windows, fixed raw anchor 335.7/597.7 Hz, both raw and KM184 > -30 dB relative to own anchor maximum; same mask and anchor before/after; group-equal summary.',
          'limitations': ['Same-session descriptive check; profiles were fitted or selected using October material.',
                          'A five-second spectrum does not isolate attack, decay or individual harmonics.',
                          'Absolute microphone quality and listener preference are not established.'],
          'summary': summary, 'window_band_rows': rows}
(OUT / 'upper-band-residual-qc.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
for device, data in summary.items():
    print(device, data['windows'], data['group_equal_before_mae_db'], data['group_equal_after_mae_db'])
    print([(b['center_hz'], b['after_median_db'], b['after_positive_fraction'], b['windows']) for b in data['bands']])
