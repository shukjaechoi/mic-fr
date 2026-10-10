"""Two controlled October upper-mid supplements for iPhone/S21.

The low/mid baseline SOS is frozen. A requested 1–2 kHz broad cut is a
comparator; the sign-aware two-section shape is derived from group-median
residuals with bounded gain. Same-session development only.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy import signal, optimize

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs/experiments/20261008-iphone-km184'
FS, N = 48000, 8192
DEVICES = ('iphone', 's21')
audit = json.loads((OUT / 'upper-band-residual-qc.json').read_text())
source_report = json.loads((OUT / 'october-four-device-eq.json').read_text())
base = {'iphone': np.asarray(json.loads((OUT / 'october-final-filter.json').read_text())['sos'], float),
        's21': np.asarray(source_report['models']['s21']['sos'], float)}
config = {'iphone': {'centers_hz': (1350, 2150), 'q': (1.3, 1.4)},
          's21': {'centers_hz': (1500, 2500), 'q': (2.0, 1.2)}}
old = json.loads((ROOT / 'docs/experiments/20261002/data/analysis.json').read_text())
maps = {m['id']: m['guitarSync'] for m in old['microphones']}
maps['iphone'] = json.loads((OUT / 'analysis.json').read_text())['sets']['guitar']['mapping']
km = np.load('/private/tmp/mic-audit-20261008/km184-guitar.npy', mmap_mode='r')
audio = {k: np.load(f'/private/tmp/mic-audit-20261008/{k}-guitar.npy', mmap_mode='r') for k in DEVICES}
freq = np.fft.rfftfreq(N, 1 / FS)
centers = 800 * 2 ** (np.arange(15) / 6)
band_masks = [(freq >= c / 2 ** (1 / 12)) & (freq < c * 2 ** (1 / 12)) for c in centers]
anchors = (335.7, 597.7)
anchor_masks = [(freq >= c / 2 ** (1 / 6)) & (freq < c * 2 ** (1 / 6)) for c in anchors]

def peak(fc, q, gain):
    a = 10 ** (gain / 40)
    w = 2 * np.pi * fc / FS
    alpha = np.sin(w) / (2 * q)
    cosine = np.cos(w)
    b = np.array([1 + alpha * a, -2 * cosine, 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * cosine, 1 - alpha / a])
    return np.r_[b / den[0], den / den[0]]

def response(sos, f):
    return 20 * np.log10(np.maximum(abs(signal.sosfreqz(sos, worN=f, fs=FS)[1]), 1e-12))

def psd(x):
    return signal.welch(np.asarray(x, float), FS, nperseg=N, noverlap=N // 2)[1]

def power(p, masks):
    return np.array([np.mean(p[m]) for m in masks])

def locate(k, t):
    ks = round(t * FS)
    m = maps[k]
    start = round(ks * m['slope'] + m.get('intercept_samples', m.get('interceptSamples', 0)))
    left = max(0, start - round(.25 * FS))
    return np.asarray(km[ks:ks + 5 * FS], float), np.asarray(audio[k][left:start + 5 * FS], float), start - left

def fit_shape(k, excluded=None):
    rows = audit['window_band_rows']
    subset = [x for x in rows if x['device'] == k and x['group'] != excluded and 1000 <= x['center_hz'] <= 3200]
    f = np.array(sorted({x['center_hz'] for x in subset}))
    target, support = [], []
    for c in f:
        rr = [x for x in subset if x['center_hz'] == c]
        group_values = [np.median([x['after_km_minus_device_db'] for x in rr if x['group'] == g])
                        for g in sorted({x['group'] for x in rr})]
        target.append(float(np.median(group_values)))
        support.append(len(group_values) / (10 if excluded else 11))
    pieces = [peak(c, q, 1) for c, q in zip(config[k]['centers_hz'], config[k]['q'])]
    matrix = np.array([response(np.array([section]), f) for section in pieces]).T
    weight = np.sqrt(support)
    reg = .4 * np.eye(2)
    gains = optimize.lsq_linear(np.vstack([matrix * weight[:, None], reg]),
                                np.r_[np.array(target) * weight, 0, 0],
                                bounds=([0, -2.5], [2.5, 0])).x
    return np.vstack([peak(c, q, g) for c, q, g in zip(config[k]['centers_hz'], config[k]['q'], gains)]), gains

profile = {}
for k in DEVICES:
    shaped, gains = fit_shape(k)
    cut = np.array([peak(1500, .9, -1.5)])
    profile[k] = {'baseline': base[k], 'broad_cut': np.vstack([base[k], cut]),
                  'shaped': np.vstack([base[k], shaped]), 'supplement': shaped,
                  'shaped_gains_db': gains.tolist(),
                  'shaped_loo': {g: np.vstack([base[k], fit_shape(k, g)[0]])
                                 for g in sorted({w['group'] for w in source_report['scores'][k]})}}

rows = []
for k in DEVICES:
    for original in source_report['scores'][k]:
        t = original['start_s']
        ref, raw_context, off = locate(k, t)
        raw = raw_context[off:]
        if len(ref) != 5 * FS or len(raw) != 5 * FS:
            continue
        rp, xp = psd(ref), psd(raw)
        rb, xb = power(rp, band_masks), power(xp, band_masks)
        ra, xa = power(rp, anchor_masks), power(xp, anchor_masks)
        mask = (10 * np.log10(np.maximum(rb / np.max(ra), 1e-30)) >= -30) & \
               (10 * np.log10(np.maximum(xb / np.max(xa), 1e-30)) >= -30)
        mask &= (centers >= 1000) & (centers <= 3200)
        if not mask.any():
            continue
        anchor = float(np.median(10 * np.log10(np.maximum(ra / xa, 1e-30))))
        results = {}
        for variant in ('baseline', 'broad_cut', 'shaped', 'shaped_loo'):
            sos = profile[k]['shaped_loo'][original['group']] if variant == 'shaped_loo' else profile[k][variant]
            filtered = signal.sosfilt(sos, raw_context)[off:]
            yp = power(psd(filtered), band_masks)
            residual = 10 * np.log10(np.maximum(rb / yp, 1e-30)) - anchor
            results[variant] = {'mae_db': float(np.mean(abs(residual[mask]))),
                                'peak_dbfs': float(20 * np.log10(max(np.max(abs(filtered)), 1e-12))),
                                'signed_residual_db': [round(float(v), 3) if good else None for v, good in zip(residual, mask)]}
        rows.append({'device': k, 'group': original['group'], 'start_s': t,
                     'valid_bands': int(sum(mask)), 'variants': results})

def aggregate(k):
    rr = [x for x in rows if x['device'] == k]
    groups = sorted({x['group'] for x in rr})
    by_group = {}
    for g in groups:
        part = [x for x in rr if x['group'] == g]
        by_group[g] = {'windows': len(part), **{v: round(float(np.mean([x['variants'][v]['mae_db'] for x in part])), 3)
                                                   for v in ('baseline', 'broad_cut', 'shaped')},
                       'broad_cut_worse_windows': sum(x['variants']['broad_cut']['mae_db'] > x['variants']['baseline']['mae_db'] + .5 for x in part),
                       'shaped_worse_windows': sum(x['variants']['shaped']['mae_db'] > x['variants']['baseline']['mae_db'] + .5 for x in part),
                       'shaped_loo': round(float(np.mean([x['variants']['shaped_loo']['mae_db'] for x in part])), 3)}
    return {'windows': len(rr), 'group_equal_mae_db': {v: round(float(np.mean([by_group[g][v] for g in groups])), 3)
                                                       for v in ('baseline', 'broad_cut', 'shaped', 'shaped_loo')},
            'worse_gt_0_5db': {v: sum(x['variants'][v]['mae_db'] > x['variants']['baseline']['mae_db'] + .5 for x in rr)
                                for v in ('broad_cut', 'shaped')},
            'max_window_peak_dbfs': {v: round(max(x['variants'][v]['peak_dbfs'] for x in rr), 3)
                                     for v in ('baseline', 'broad_cut', 'shaped')},
            'groups': by_group}

response_hz = np.geomspace(300, 6000, 240)
models = {}
for k in DEVICES:
    models[k] = {'shaped_section_centers_hz': config[k]['centers_hz'], 'shaped_section_q': config[k]['q'],
                 'shaped_section_gains_db': [round(float(v), 4) for v in profile[k]['shaped_gains_db']],
                 'broad_cut': {'center_hz': 1500, 'q': .9, 'gain_db': -1.5},
                 'sos': {v: profile[k][v].tolist() for v in ('baseline', 'broad_cut', 'shaped')},
                 'sha256': {v: hashlib.sha256(np.asarray(profile[k][v], dtype='<f8').tobytes()).hexdigest()
                            for v in ('baseline', 'broad_cut', 'shaped')},
                 'response_db': {v: np.round(response(profile[k][v], response_hz), 4).tolist()
                                 for v in ('baseline', 'broad_cut', 'shaped')}}

out = {'scope': 'October same-session development, iPhone and S21 only; frozen baseline plus upper-mid supplement',
       'method': 'Group-median signed residual on 1/6-octave centers 1–3.2kHz; two fixed peaking sections and ridge .4, positive gain bounded 0–2.5dB and cut -2.5–0dB. Broad-cut comparator is -1.5dB at 1.5kHz Q .9. Both are actual causal SOS, filtered with 250ms preceding audio. Evaluation uses same raw-observability mask and raw-derived anchor for all variants.',
       'limitations': ['Shaped gains were fitted on these same October groups; this is an in-session diagnostic, not an independent generalization score.', 'The fixed baseline was already chosen or fitted using October data.', 'Five-second band error does not isolate percussion attack or listener preference.'],
       'response_hz': response_hz.tolist(), 'models': models,
       'summary': {k: aggregate(k) for k in DEVICES}, 'windows': rows}
(OUT / 'upper-mid-supplement.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
for k in DEVICES:
    print(k, models[k]['shaped_section_gains_db'], out['summary'][k], flush=True)
