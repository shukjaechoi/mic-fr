"""Audit previously omitted 6–10 kHz brightness and one S21 shelf candidate.

No new iPhone cut is inferred from a listener adjective: spectrum and reliable
percussion/chord observations must support the direction first.
"""
from pathlib import Path
import json, hashlib
import numpy as np
import soundfile as sf
from scipy import signal, ndimage
from PIL import Image
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / 'docs/experiments/20261008-iphone-km184'
OUT = ROOT / 'docs/experiments/20261008-iphone-km184-public'
FS = 48000
prior = json.loads((LOCAL / 'speaker-guitar-confidence-eq.json').read_text())
background = json.loads((LOCAL / 'guitar-background-v2.json').read_text())
old = json.loads((ROOT / 'docs/experiments/20261002/data/analysis.json').read_text())
maps = {m['id']: m['guitarSync'] for m in old['microphones']}
smaps = {m['id']: m['speakerSync'] for m in old['microphones']}
an = json.loads((LOCAL / 'analysis.json').read_text())
maps['iphone'] = an['sets']['guitar']['mapping']; smaps['iphone'] = an['sets']['speaker']['mapping']
audio = {d: np.load(f'/private/tmp/mic-audit-20261008/{d}-guitar.npy', mmap_mode='r') for d in ('km184', 'iphone', 's21')}
speaker_audio = {d: np.load(f'/private/tmp/mic-audit-20261008/{d}-speaker.npy', mmap_mode='r') for d in ('km184', 'iphone')}
speaker_audio['s21'], sr = sf.read('/Users/sjchoi/dev/rec/261002-iphone,h1,km184,s21,d10/s21 speaker.wav')
assert sr == FS
band_bounds = [(400, 800), (800, 1600), (1600, 3200), (3200, 6000), (6000, 10000)]
def mapped(m, t): return round(t * FS * m['slope'] + m.get('interceptSamples', m.get('intercept_samples', 0)))
def psd(x): return signal.welch(np.asarray(x, float), FS, nperseg=8192, noverlap=4096)[1]
freq = np.fft.rfftfreq(8192, 1 / FS)
masks = [(freq >= lo) & (freq < hi) for lo, hi in band_bounds]
def powers(p): return np.array([np.sum(p[m]) for m in masks])
def ratio_db(p):
    v = powers(p)
    return 10 * np.log10(np.maximum(v / (v[0] + v[1]), 1e-30))
def context(d, t):
    start = mapped(maps[d], t); left = max(0, start - FS // 4)
    return np.asarray(audio[d][left:start + 5 * FS], float), start - left
def response(sos, hz): return 20 * np.log10(np.maximum(abs(signal.sosfreqz(sos, worN=hz, fs=FS)[1]), 1e-12))
def shelf(fc, gain):
    # RBJ high shelf with slope S=1; a single stable causal biquad.
    a = 10 ** (gain / 40); w = 2 * np.pi * fc / FS; c = np.cos(w)
    alpha = np.sin(w) / np.sqrt(2); z = 2 * np.sqrt(a) * alpha
    b = a * np.array([(a + 1) + (a - 1) * c + z, -2 * ((a - 1) + (a + 1) * c), (a + 1) + (a - 1) * c - z])
    den = np.array([(a + 1) - (a - 1) * c + z, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - z])
    return np.r_[b / den[0], den / den[0]]
floor = {d: np.array(background['median_psd'][d]) for d in ('km184', 'iphone')}
floor['s21'] = np.median([psd(audio['s21'][mapped(maps['s21'], q['start_s']):mapped(maps['s21'], q['start_s'] + 2)]) for q in background['selected']], axis=0)
floor_power = {d: powers(p) for d, p in floor.items()}
current = {d: np.array(prior['models'][d]['sos']) for d in ('iphone', 's21')}
rows = {d: [] for d in current}
for d in current:
    for group, rr in prior['windows'][d].items():
        for r in rr:
            t = r['start_s']; ref = np.asarray(audio['km184'][round(t * FS):round((t + 5) * FS)], float)
            x, off = context(d, t)
            if len(ref) != 5 * FS or len(x[off:]) != 5 * FS: continue
            rp, xp = psd(ref), psd(x[off:]); kp, dp = powers(rp), powers(xp)
            snr = np.minimum(10 * np.log10(np.maximum(kp / floor_power['km184'], 1e-30)), 10 * np.log10(np.maximum(dp / floor_power[d], 1e-30)))
            denominator_snr = min(10 * np.log10((kp[0] + kp[1]) / sum(floor_power['km184'][:2])),
                                  10 * np.log10((dp[0] + dp[1]) / sum(floor_power[d][:2])))
            snr = np.minimum(snr, denominator_snr)
            strength = np.minimum(10 * np.log10(kp / max(kp)), 10 * np.log10(dp / max(dp)))
            weight = np.clip((snr - 10) / 15, 0, 1) * (.2 + .8 * np.clip((strength + 36) / 24, 0, 1))
            yp = psd(signal.sosfilt(current[d], x)[off:])
            residual = ratio_db(yp) - ratio_db(rp)
            rows[d].append({'group': group, 'start_s': t, 'joint_snr_db': snr.tolist(), 'weight': weight.tolist(), 'current_relative_excess_db': residual.tolist()})

speaker = {d: {} for d in current}
for name, a, b in [('pink', 252, 262), ('white', 219, 232), ('log', 525.2, 543.9)]:
    p = {d: psd(v[mapped(smaps[d], a):mapped(smaps[d], b)]) for d, v in speaker_audio.items()}
    anchor_mask = (freq >= 300) & (freq < 700)
    for d in current:
        gain = 10 * np.log10(np.sum(p['km184'][anchor_mask]) / np.sum(p[d][anchor_mask]))
        target = 10 * np.log10(powers(p['km184']) / powers(p[d])) - gain
        speaker[d][name] = target.tolist()

# One pre-defined broad shelf; gain estimated from reliable cross-technique
# medians and bounded at -6 dB. No per-note or listener-selected notches.
selected = [r for r in rows['s21'] if r['weight'][4] >= .25]
groups = sorted(set(r['group'] for r in selected))
group_excess = {g: float(np.median([r['current_relative_excess_db'][4] for r in selected if r['group'] == g])) for g in groups}
excess = float(np.median(list(group_excess.values())))
source_high = [speaker['s21'][name][4] for name in ('pink', 'white', 'log')]
assert len(groups) >= 3 and excess > 0 and all(x < 0 for x in source_high)
unit = float(np.mean(response(np.array([shelf(6000, -1)]), np.geomspace(6000, 10000, 80))))
gain = float(np.clip(excess / unit, -6, 0))
extended = {'s21': np.vstack([current['s21'], shelf(6000, gain)])}
for r in rows['s21']:
    t = r['start_s']; rp = psd(audio['km184'][round(t * FS):round((t + 5) * FS)])
    x, off = context('s21', t)
    r['extended_relative_excess_db'] = (ratio_db(psd(signal.sosfilt(extended['s21'], x)[off:])) - ratio_db(rp)).tolist()

summary = {}
for d in current:
    reliable = [r for r in rows[d] if r['weight'][4] >= .25]
    summary[d] = {'reliable_high_windows': len(reliable), 'candidate_windows': len(rows[d]),
                  'high_positive_excess_fraction': sum(r['current_relative_excess_db'][4] > 0 for r in reliable) / len(reliable) if reliable else None,
                  'high_excess_median_db': float(np.median([r['current_relative_excess_db'][4] for r in reliable])) if reliable else None}
    if d == 's21':
        summary[d].update({'high_extended_excess_median_db': float(np.median([r['extended_relative_excess_db'][4] for r in reliable])),
                          'high_absolute_error_worse_gt_0_5db': sum(abs(r['extended_relative_excess_db'][4]) > abs(r['current_relative_excess_db'][4]) + .5 for r in reliable),
                          'group_medians_before_db': group_excess,
                          'group_medians_after_db': {g: float(np.median([r['extended_relative_excess_db'][4] for r in reliable if r['group'] == g])) for g in groups}})

models = {'s21': {'center_hz': 6000, 'slope': 1, 'gain_db': gain, 'sos': extended['s21'].tolist(),
                  'output_gain_db': -6.5, 'sha256': hashlib.sha256(np.asarray(extended['s21'], dtype='<f8').tobytes()).hexdigest()}}
# Sample and 4x oversampling peaks with state retained on the complete S21 file.
zi = np.zeros((len(extended['s21']), 2)); peak = 0.; oversampled = 0.
for st in range(0, len(audio['s21']), 10 * FS):
    y, zi = signal.sosfilt(extended['s21'], np.asarray(audio['s21'][st:st + 10 * FS], float), zi=zi)
    peak = max(peak, float(np.max(abs(y))))
    # Context overlap on each block excludes resampler padding at its edges.
    begin = max(0, st - FS // 4); end = min(len(audio['s21']), st + 10 * FS + FS // 4)
    ctx = signal.sosfilt(extended['s21'], np.asarray(audio['s21'][begin:end], float))
    ov = signal.resample_poly(ctx, 4, 1)[(st - begin) * 4:(min(st + 10 * FS, len(audio['s21'])) - begin) * 4]
    oversampled = max(oversampled, float(np.max(abs(ov))))
models['s21'].update({'full_sample_peak_dbfs_before_output_gain': 20 * np.log10(peak), 'approx_4x_peak_dbfs_before_output_gain': 20 * np.log10(oversampled)})

freqgrid = np.geomspace(12000, 80, 280)
def spec(x):
    f, t, z = signal.stft(x, FS, nperseg=2048, noverlap=1536, boundary=None)
    fi, ti = np.meshgrid(freqgrid / (f[1] - f[0]), (np.linspace(0, 5, 470) - t[0]) / (t[1] - t[0]), indexing='ij')
    return ndimage.map_coordinates(20 * np.log10(np.maximum(abs(z), 1e-8)), [fi, ti], order=1, mode='nearest')
manifest = []
for group, t, label in [('arpeggio', 100, '아르페지오'), ('open_notes', 418, '개방현 단음'), ('complex', 1704, '복합 화음'), ('percussion', 1792, '퍼커션')]:
    ref = np.asarray(audio['km184'][t * FS:(t + 5) * FS], float)
    for d in current:
        x, off = context(d, t); raw = x[off:]; rp, xp = psd(ref), psd(raw)
        anchor = np.median([10 * np.log10(np.mean(rp[(freq >= c / 2 ** (1 / 6)) & (freq < c * 2 ** (1 / 6))]) / np.mean(xp[(freq >= c / 2 ** (1 / 6)) & (freq < c * 2 ** (1 / 6))])) for c in (335.7, 597.7)])
        waves = {'km184': ref, 'raw': raw, 'current': signal.sosfilt(current[d], x)[off:]}
        if d == 's21': waves['extended'] = signal.sosfilt(extended[d], x)[off:]
        for k in waves:
            if k != 'km184': waves[k] *= 10 ** (anchor / 20)
        scale = min(1, .8 / max(float(np.max(abs(v))) for v in waves.values()))
        waves = {k: v * scale for k, v in waves.items()}
        spectra = {k: spec(v) for k, v in waves.items()}; top = max(float(v.max()) for v in spectra.values())
        refspec = ndimage.gaussian_filter(spectra['km184'], (2, 1.5)); media = {}
        for k, v in waves.items():
            stem = f'brightness-{group}-{t}-{d}-{k}'
            sf.write(OUT / (stem + '.wav'), v, FS, subtype='PCM_16')
            colors = colormaps['magma'](np.clip((spectra[k] - top + 72) / 72, 0, 1))[:, :, :3]
            Image.fromarray(np.uint8(colors * 255)).save(OUT / (stem + '.png'), optimize=True)
            media[k] = {'audio': stem + '.wav', 'spectrogram': stem + '.png'}
            if k != 'km184':
                diff = ndimage.gaussian_filter(spectra[k], (2, 1.5)) - refspec
                observed = (refspec > top - 58) & (spectra[k] > top - 58)
                colors = colormaps['RdBu_r'](np.clip((diff + 12) / 24, 0, 1))[:, :, :3]; colors[~observed] = [.88, .90, .91]
                Image.fromarray(np.uint8(colors * 255)).save(OUT / (stem + '-error.png'), optimize=True)
                media[k]['residual'] = stem + '-error.png'
        snr = np.minimum(10 * np.log10(powers(rp) / floor_power['km184']), 10 * np.log10(powers(xp) / floor_power[d]))
        kp, dp = powers(rp), powers(xp)
        snr = np.minimum(snr, min(10 * np.log10(sum(kp[:2]) / sum(floor_power['km184'][:2])),
                                 10 * np.log10(sum(dp[:2]) / sum(floor_power[d][:2]))))
        metrics = {k: (ratio_db(psd(v)) - ratio_db(psd(waves['km184']))).round(3).tolist() for k, v in waves.items() if k != 'km184'}
        manifest.append({'group': group, 'label': label, 'start_s': t, 'device': d, 'media': media,
                         'joint_snr_db': snr.round(2).tolist(), 'relative_band_excess_db': metrics})

hz = np.geomspace(100, 16000, 320)
report = {'scope': '2026-10-02 same-session diagnostic and one bounded S21 high-shelf trial',
          'user_observation': 'Prior trial improved somewhat but corrected sound still feels brighter than KM184.',
          'method': 'Band-integrated spectral ratios relative to 400–1600Hz; the same raw-derived joint background SNR and energy importance are used before/after. S21 gain derived from medians of reliably observed technique groups and bounded to -6dB at 6kHz S=1; pink/white/log independently considered as correlated sign support. No iPhone high cut: wide-band chords do not support high-frequency excess. Existing five-second window population remains selected by the earlier mid-band pipeline, so coverage is incomplete.',
          'summary': summary, 'speaker_km_minus_device_db': speaker, 'band_bounds_hz': band_bounds,
          'models': models, 'windows': rows, 'clips': manifest, 'spectrogram_frequency_top_hz': 12000,
          'response_hz': hz.tolist(), 'response_db': {d: {'current': response(current[d], hz).round(3).tolist(), **({'extended': response(extended[d], hz).round(3).tolist()} if d == 's21' else {})} for d in current}}
(LOCAL / 'brightness-extension.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
(OUT / 'brightness-extension-browser.json').write_text(json.dumps(report, ensure_ascii=False, separators=(',', ':')) + '\n')
print(json.dumps({'summary': summary, 'model': models}, ensure_ascii=False, indent=2))
