"""Build a broad, minimum-phase biquad EQ alternative to iphone2km184 FIR.

Run after analyze.py and build_iphone_eq.py with NumPy available.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from analyze import FS, lufs, spectrum, smooth_log, content_coverage
from build_iphone_eq import read_wav, write_wav

DATA = Path(__file__).parent / 'docs' / 'data'
ANALYSIS = DATA / 'analysis.json'
OUT = DATA / 'iphone2km184-iir.wav'
ID = 'iphone2km184-iir'
# Broad peaking biquads: no narrow inverse notches or sharp high-frequency EQ.
BANDS = [(100, .75), (180, .9), (320, .95), (650, .9), (1300, .85), (2800, 1.0)]


def biquad(frequency, q, gain_db):
    omega = 2 * np.pi * frequency / FS
    alpha = np.sin(omega) / (2 * q)
    a = 10 ** (gain_db / 40)
    cosine = np.cos(omega)
    a0 = 1 + alpha / a
    return np.array([(1 + alpha * a) / a0, -2 * cosine / a0,
                     (1 - alpha * a) / a0, -2 * cosine / a0,
                     (1 - alpha / a) / a0])


def response_db(frequencies, gains):
    z = np.exp(-2j * np.pi * frequencies / FS)
    result = np.zeros_like(frequencies, dtype=float)
    for (frequency, q), gain in zip(BANDS, gains):
        b0, b1, b2, a1, a2 = biquad(frequency, q, gain)
        h = (b0 + b1 * z + b2 * z * z) / (1 + a1 * z + a2 * z * z)
        result += 20 * np.log10(np.maximum(np.abs(h), 1e-12))
    return result


def fit_bands(frequencies, curve, coverage):
    # Only broad, programme-supported variation is worth matching. Smooth the
    # original 1/6-octave estimate, then taper the desired EQ to zero outside
    # the 70 Hz–4 kHz area; penalize high-frequency changes and large gains.
    kernel_x = np.arange(-9, 10)
    kernel = np.exp(-.5 * (kernel_x / 3) ** 2)
    kernel /= kernel.sum()
    target = np.convolve(np.pad(-np.asarray(curve), (9, 9), mode='edge'), kernel, mode='valid')
    low = np.clip((frequencies - 60) / 35, 0, 1)
    high = np.clip((5000 - frequencies) / 1500, 0, 1)
    target = np.clip(target, -6, 8) * (low * low * (3 - 2 * low)) * (high * high * (3 - 2 * high))
    weights = .2 + .8 * np.asarray(coverage)
    weights[frequencies >= 5000] = .8
    gains = np.zeros(len(BANDS))

    def score(candidate):
        error = response_db(frequencies, candidate) - target
        return float(np.mean(weights * error * error) + .012 * np.sum(candidate * candidate))

    # Deterministic coordinate search; only six gain parameters are fitted.
    for step in [1, .5, .2, .1]:
        for _ in range(5):
            changed = False
            for i in range(len(gains)):
                best_gain, best_score = gains[i], score(gains)
                for value in np.arange(-8, 8.01, step):
                    trial = gains.copy()
                    trial[i] = value
                    trial_score = score(trial)
                    if trial_score < best_score:
                        best_gain, best_score = value, trial_score
                if abs(best_gain - gains[i]) > 1e-8:
                    gains[i] = best_gain
                    changed = True
            if not changed:
                break
    return gains


def apply_biquads(x, gains):
    y = x.copy()
    for (frequency, q), gain in zip(BANDS, gains):
        b0, b1, b2, a1, a2 = biquad(frequency, q, gain)
        z1 = z2 = 0.0
        for i, sample in enumerate(y):
            output = b0 * sample + z1
            z1 = b1 * sample - a1 * output + z2
            z2 = b2 * sample - a2 * output
            y[i] = output
    return y


def main():
    payload = json.loads(ANALYSIS.read_text())
    payload['microphones'] = [m for m in payload['microphones'] if m['id'] != ID]
    iphone = next(m for m in payload['microphones'] if m['id'] == 'iphone')
    centers = np.asarray(payload['frequencies'])
    gains = fit_bands(centers, iphone['curve'], iphone['coverage'])
    x = read_wav(DATA / 'iphone.wav')
    ref = read_wav(DATA / 'km184.wav')
    assert len(x) == len(ref)
    y = apply_biquads(x, gains)
    gain_db = float(payload['targetLufs']) - lufs(y)
    y *= 10 ** (gain_db / 20)
    peak = float(np.max(np.abs(y)))
    if peak >= .98:
        raise RuntimeError(f'LUFS match would clip ({peak:.3f}); reduce boosts and retry')
    write_wav(OUT, y)
    y = read_wav(OUT)  # Measure the actual 16-bit deliverable.
    spec, power, active, freq = spectrum(y)
    ref_spec, _, _, _ = spectrum(ref)
    ratio = 10 * np.log10(np.maximum(spec, 1e-20) / np.maximum(ref_spec, 1e-20))
    ratio -= np.median(ratio[(freq >= 500) & (freq <= 2000)])
    bands = {key: round(float(np.median(ratio[(freq >= lo) & (freq < hi)])), 1)
             for key, lo, hi in [('bass', 80, 250), ('lowMid', 250, 500),
                                 ('presence', 2000, 5000), ('air', 8000, 16000)]}
    mid = (freq >= 500) & (freq < 2000)
    high = (freq >= 8000) & (freq < 16000)
    high_floor = 10 * np.log10(np.maximum(spec[high].mean(), 1e-20) / np.maximum(spec[mid].mean(), 1e-20))
    coverage = content_coverage(power, active, freq, centers)
    row = dict(id=ID, label='iphone2km184-IIR', derivedFrom='iphone',
               originalLufs=iphone['originalLufs'], gainDb=round(gain_db, 2),
               processedLufs=round(lufs(y), 2), lagMs=iphone['lagMs'],
               syncSpreadMs=iphone['syncSpreadMs'], polarityInverted=False,
               peakDbfs=round(20 * np.log10(np.max(np.abs(y))), 1), bands=bands,
               highToMidDb=round(float(high_floor), 1),
               curve=[round(v, 2) if v is not None else None for v in smooth_log(freq, ratio, centers)],
               coverage=[round(float(v), 3) for v in coverage], audio='data/iphone2km184-iir.wav',
               eqBands=[dict(hz=frequency, q=q, gainDb=round(float(gain), 1))
                        for (frequency, q), gain in zip(BANDS, gains)])
    index = next(i for i, m in enumerate(payload['microphones']) if m['id'] == 'iphone2km184')
    payload['microphones'].insert(index + 1, row)
    payload['iirNote'] = ('iphone2km184-IIR uses six broad, cascaded second-order peaking EQ sections '
                          'fitted conservatively to the same programme-derived iPhone/KM184 spectrum. '
                          'It is minimum-phase, avoids linear-phase FIR pre-ringing, and is LUFS-matched. '
                          'It is not a calibrated KM184 emulation; phase, noise, distortion and directivity remain different.')
    ANALYSIS.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print('saved', OUT, 'LUFS', row['processedLufs'], 'peak dBFS', row['peakDbfs'])
    print('EQ', row['eqBands'])
    print('bands', bands)


if __name__ == '__main__':
    main()
