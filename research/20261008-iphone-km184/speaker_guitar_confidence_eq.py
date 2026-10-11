"""Same-session speaker-prior/guitar-confidence mid-high EQ experiment.

The speaker recordings are a prior, not a direct-sound or guitar transfer
function. A fixed low-frequency base is retained; the older upper EQ is
replaced. All reported scores reuse the same source-derived masks.
"""
from pathlib import Path
import hashlib, json
import numpy as np
import soundfile as sf
from scipy import optimize, signal

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / 'docs/experiments/20261008-iphone-km184'
PUBLIC = ROOT / 'docs/experiments/20261008-iphone-km184-public'
FS, N = 48000, 8192
DEV = ('iphone', 's21')
old = json.loads((ROOT / 'docs/experiments/20261002/data/analysis.json').read_text())
gmaps = {m['id']: m['guitarSync'] for m in old['microphones']}
smaps = {m['id']: m['speakerSync'] for m in old['microphones']}
october = json.loads((LOCAL / 'analysis.json').read_text())
gmaps['iphone'] = october['sets']['guitar']['mapping']
smaps['iphone'] = october['sets']['speaker']['mapping']
previous = json.loads((PUBLIC / 'upper-mid-supplement-browser.json').read_text())
windows = json.loads((LOCAL / 'october-four-device-eq.json').read_text())['scores']
background = json.loads((LOCAL / 'guitar-background-v2.json').read_text())
gaudio = {d: np.load(f'/private/tmp/mic-audit-20261008/{d}-guitar.npy', mmap_mode='r') for d in ('km184', *DEV)}
saudio = {d: np.load(f'/private/tmp/mic-audit-20261008/{d}-speaker.npy', mmap_mode='r') for d in ('km184', 'iphone')}
with sf.SoundFile('/Users/sjchoi/dev/rec/261002-iphone,h1,km184,s21,d10/s21 speaker.wav') as f:
    saudio['s21'] = f.read(dtype='float32')
f = np.fft.rfftfreq(N, 1 / FS)
centers = np.array([800, 1008, 1270, 1600, 2016, 2540, 3200, 4032, 5080.])
bmask = [(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6)) for c in centers]
amask = [(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6)) for c in (335.7, 597.7)]
def psd(x): return signal.welch(np.asarray(x, float), FS, nperseg=N, noverlap=N // 2)[1]
def bands(p): return np.array([np.mean(p[m]) for m in bmask])
def anchor(k, x): return float(np.median(10 * np.log10(np.maximum(np.array([np.mean(k[m]) / np.mean(x[m]) for m in amask]), 1e-30))))
def mapped(m, seconds): return round(seconds * FS * m['slope'] + m.get('interceptSamples', m.get('intercept_samples', 0)))
def peak(fc, q, gain):
    a = 10 ** (gain / 40); w = 2 * np.pi * fc / FS; alpha = np.sin(w) / (2 * q); c = np.cos(w)
    b = np.array([1 + alpha * a, -2 * c, 1 - alpha * a]); z = np.array([1 + alpha / a, -2 * c, 1 - alpha / a])
    return np.r_[b / z[0], z / z[0]]
def response(sos, hz): return 20 * np.log10(np.maximum(abs(signal.sosfreqz(sos, worN=hz, fs=FS)[1]), 1e-12))

# The same stationary/sweep passage is cut from each simultaneous speaker recording.
# Linear sweep is retained in QC but not the fit because it contradicts all other
# sources by >10 dB at many centers under this whole-passage PSD estimator.
segments = {'pink': (252, 262), 'white': (219, 232), 'log': (525.2, 543.9), 'linear_qc': (548.1, 566.6)}
speaker = {d: {} for d in DEV}
for name, (a, b) in segments.items():
    spectral = {}
    for d in ('km184', *DEV):
        x = np.asarray(saudio[d][mapped(smaps[d], a):mapped(smaps[d], b)], float)
        spectral[d] = psd(x)
    for d in DEV:
        shape = 10 * np.log10(np.maximum(bands(spectral['km184']) / bands(spectral[d]), 1e-30))
        shape -= anchor(spectral['km184'], spectral[d])
        speaker[d][name] = shape

floor = {d: np.asarray(background['median_psd'][d], float) for d in ('km184', 'iphone')}
for d in ('s21',):
    estimates = []
    for quiet in background['selected']:
        t = quiet['start_s']; x = np.asarray(gaudio[d][mapped(gmaps[d], t):mapped(gmaps[d], t + 2)], float)
        estimates.append(psd(x))
    floor[d] = np.median(estimates, axis=0)
floor_bands = {d: bands(p) for d, p in floor.items()}
guitar = {d: [] for d in DEV}
for d in DEV:
    for w in windows[d]:
        t = float(w['start_s']); r0 = round(t * FS); v0 = mapped(gmaps[d], t)
        ref = np.asarray(gaudio['km184'][r0:r0 + 5 * FS], float)
        raw = np.asarray(gaudio[d][v0:v0 + 5 * FS], float)
        if len(ref) != 5 * FS or len(raw) != 5 * FS: continue
        rp, xp = psd(ref), psd(raw); rb, xb = bands(rp), bands(xp)
        snr = np.minimum(10 * np.log10(np.maximum(rb / floor_bands['km184'], 1e-30)),
                         10 * np.log10(np.maximum(xb / floor_bands[d], 1e-30)))
        # Observability and relevance are distinct. Energy is integrated over band
        # width, so a wide-band percussion event is not penalized by mean PSD.
        width = centers * (2 ** (1 / 6) - 2 ** (-1 / 6))
        kr = 10 * np.log10(np.maximum(rb * width / max(rb * width), 1e-30))
        xr = 10 * np.log10(np.maximum(xb * width / max(xb * width), 1e-30))
        relevance = .2 + .8 * np.clip((np.minimum(kr, xr) + 36) / 24, 0, 1)
        confidence = np.clip((snr - 10) / 15, 0, 1) * relevance
        gain = anchor(rp, xp)
        residual = 10 * np.log10(np.maximum(rb / xb, 1e-30)) - gain
        guitar[d].append({'group': w['group'], 'start_s': t, 'confidence': confidence,
                          'snr_db': snr, 'relevance': relevance, 'residual_db': residual,
                          'anchor_db': gain, 'ref_band': rb, 'raw_band': xb})

# Three reference sources are not independent measurements; spread explicitly
# reduces their weight.  A band enters only if >=3 technique groups observe it
# and the median guitar direction agrees with the speaker prior.
configs = {'iphone': [(1600, .8), (3200, .8)], 's21': [(1600, .75), (4200, .7)]}
old_base = {d: np.asarray(previous['models'][d]['sos']['shaped'], float) for d in DEV}
# The iPhone replacement trial was worse on 95/135 windows, including chords
# and percussion. Keep its existing EQ as the fixed base; seek only a supported
# residual supplement. S21 retains its original low sections.
low_base = {'iphone': old_base['iphone'],
            's21': np.asarray(json.loads((LOCAL / 'october-four-device-eq.json').read_text())['models']['s21']['sos'][:3], float)}
models, evidence = {}, {}
for d in DEV:
    source = np.stack([speaker[d][s] for s in ('pink', 'white', 'log')])
    s_target = np.median(source, axis=0) - response(low_base[d], centers)
    s_spread = np.max(source, axis=0) - np.min(source, axis=0)
    groups = sorted(set(r['group'] for r in guitar[d])); gv, gc = [], []
    for g in groups:
        rr = [r for r in guitar[d] if r['group'] == g]
        gv.append([np.median([r['residual_db'][i] - response(low_base[d], centers)[i] for r in rr if r['confidence'][i] >= .25])
                   if any(r['confidence'][i] >= .25 for r in rr) else np.nan for i in range(len(centers))])
        gc.append([np.median([r['confidence'][i] for r in rr]) for i in range(len(centers))])
    gv, gc = np.array(gv), np.array(gc)
    support = np.sum(np.isfinite(gv), axis=0)
    g_target = np.nanmedian(gv, axis=0)
    agreement = np.where(np.isfinite(g_target), np.sign(g_target) == np.sign(s_target), False)
    # Speaker reference is a prior; the actual guitar estimate dominates only
    # in strongly observed, cross-technique bands with the same sign.
    reliability = np.clip((6 - s_spread) / 5, 0, 1) * np.clip(support / 5, 0, 1) * agreement
    target = np.where(agreement, .45 * s_target + .55 * np.nan_to_num(g_target), 0)
    target = np.clip(target, -4, 4)
    effective_weight = reliability * np.clip(np.nanmean(gc, axis=0), 0, 1)
    # Preserve only broad, interpretable sections; no per-harmonic notch fitting.
    unit = np.array([response(np.array([peak(c, q, 1)]), centers) for c, q in configs[d]]).T
    A = np.vstack([unit * np.sqrt(effective_weight[:, None]), .35 * np.eye(len(configs[d]))])
    b = np.r_[target * np.sqrt(effective_weight), np.zeros(len(configs[d]))]
    sol = optimize.lsq_linear(A, b, bounds=(-4, 4)).x
    new = np.vstack([low_base[d], *[peak(c, q, g) for (c, q), g in zip(configs[d], sol)]])
    models[d] = {'sos': new.tolist(), 'sha256': hashlib.sha256(np.asarray(new, dtype='<f8').tobytes()).hexdigest(),
                 'sections': [{'center_hz': c, 'q': q, 'gain_db': round(float(g), 4)} for (c, q), g in zip(configs[d], sol)]}
    evidence[d] = {'speaker_km_minus_device_db': {k: v.round(3).tolist() for k, v in speaker[d].items()},
                   'speaker_three_source_spread_db': s_spread.round(3).tolist(),
                   'speaker_km_minus_fixed_base_db': s_target.round(3).tolist(),
                   'guitar_group_median_km_minus_low_base_db': np.round(g_target, 3).tolist(),
                   'guitar_support_groups': support.tolist(), 'same_direction': agreement.tolist(),
                   'fit_weight': effective_weight.round(3).tolist(), 'target_db': target.round(3).tolist(),
                   'group_confidence': {g: np.round(gc[j], 3).tolist() for j, g in enumerate(groups)}}

# Score whole phrases with a common raw-derived level and the same raw-derived
# confidence mask. Preserve preceding audio for causal filter state.
scores = {d: {} for d in DEV}
for d in DEV:
    for r in guitar[d]:
        t = r['start_s']; start = mapped(gmaps[d], t); left = max(0, start - FS // 4); off = start - left
        x = np.asarray(gaudio[d][left:start + 5 * FS], float)
        versions = {'raw': x[off:], 'existing': signal.sosfilt(old_base[d], x)[off:],
                    'new': signal.sosfilt(np.asarray(models[d]['sos']), x)[off:]}
        score = {}
        for name, y in versions.items():
            py = bands(psd(y)); err = 10 * np.log10(np.maximum(r['ref_band'] / py, 1e-30)) - r['anchor_db']
            weight = r['confidence']; valid = weight >= .15
            score[name] = float(np.average(abs(err[valid]), weights=weight[valid])) if valid.any() else None
        scores[d].setdefault(r['group'], []).append({'start_s': t, 'weight_sum': float(sum(r['confidence'])), 'mae_db': score})

summary = {}
for d in DEV:
    by_group = {}
    for g, rr in scores[d].items():
        eligible = [x for x in rr if all(v is not None for v in x['mae_db'].values())]
        by_group[g] = {'windows': len(eligible), 'mean_mae_db': {v: round(float(np.mean([x['mae_db'][v] for x in eligible])), 3) if eligible else None for v in ('raw', 'existing', 'new')},
                       'new_worse_than_existing_gt_0_5db': sum(x['mae_db']['new'] > x['mae_db']['existing'] + .5 for x in eligible)}
    summary[d] = {'groups': by_group, 'equal_group_mae_db': {v: round(float(np.mean([x['mean_mae_db'][v] for x in by_group.values() if x['mean_mae_db'][v] is not None])), 3) for v in ('raw', 'existing', 'new')},
                  'worse_windows': sum(x['new_worse_than_existing_gt_0_5db'] for x in by_group.values()),
                  'valid_windows': sum(x['windows'] for x in by_group.values())}

for d in DEV:
    # Continuous SOS state across the complete source checks the required
    # output headroom, rather than relying on isolated five-second previews.
    for name, sos in (('existing', old_base[d]), ('new', np.asarray(models[d]['sos']))):
        state = np.zeros((len(sos), 2)); maximum = 0.
        for start in range(0, len(gaudio[d]), 10 * FS):
            filtered, state = signal.sosfilt(sos, np.asarray(gaudio[d][start:start + 10 * FS], float), zi=state)
            maximum = max(maximum, float(np.max(abs(filtered))))
        summary[d].setdefault('full_recording_sample_peak_dbfs_before_output_gain', {})[name] = round(20 * np.log10(max(maximum, 1e-12)), 3)

hz = np.geomspace(100, 8000, 320)
report = {'scope': '2026-10-02 same-session development; iPhone/S21 only', 'centers_hz': centers.tolist(),
          'methods': {'speaker': 'Same-passage KM184/device 1/3-octave PSD ratios, 300–700Hz anchor. Pink, white and AudioCheck log contribute as correlated priors. AudioCheck linear is QC-only because it disagrees strongly under this estimator. Speaker geometry/room differ from guitar.',
                      'guitar': 'Five-second labeled windows; joint SNR above measured background and integrated band energy relevance determine per-technique confidence. At least three groups plus sign agreement with speaker prior are required. S21 retains its low SOS and replaces upper sections; iPhone retains its entire earlier EQ and tests two supplemental sections after a replacement trial failed.',
                      'score': 'Same guitar windows, same raw-derived 335.7/597.7Hz anchor and confidence for all versions. This is exploratory in-session evaluation, not independent proof or perceptual texture score.',
                      'peak': 'Continuous source filtered in 10-second blocks while preserving SOS state; sample peak before any output attenuation. True peak and app input headroom not checked.'},
          'evidence': evidence, 'models': models, 'summary': summary, 'windows': scores,
          'response_hz': hz.tolist(), 'response_db': {d: {v: response(s, hz).round(3).tolist() for v, s in [('low',low_base[d]),('existing',old_base[d]),('new',np.asarray(models[d]['sos']))]} for d in DEV}}
(LOCAL / 'speaker-guitar-confidence-eq.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n')
print(json.dumps({d: {'sections': models[d]['sections'], **summary[d]} for d in DEV}, ensure_ascii=False, indent=2))
