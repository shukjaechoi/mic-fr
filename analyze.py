"""Reproducible, relative comparison of the seven simultaneous microphone takes.

Requires NumPy. Run with the bundled Python or any Python with numpy installed.
No sweep/reference stimulus was recorded, so these are *relative programme spectra*,
not calibrated microphone frequency responses.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path
import numpy as np

SOURCE = Path('/Users/sjchoi/dev/mic-fr')
OUT = Path(__file__).parent / 'docs' / 'data'
OUT.mkdir(parents=True, exist_ok=True)
FS = 48000
TARGET = -20.0
NAMES = ['Beringer C-2', 'em-usb', 'iphone', 'km184', 'TLM103', 'sm57', 'sonyPCMD10']
REF = 'km184'


def read(name):
    with wave.open(str(SOURCE / f'{name}.wav'), 'rb') as f:
        assert f.getframerate() == FS and f.getsampwidth() == 2
        x = np.frombuffer(f.readframes(f.getnframes()), '<i2').reshape(-1, f.getnchannels()).astype(np.float64) / 32768
    return x.mean(axis=1)


def biquad(x, b, a):
    # Direct Form II transposed; 48 kHz BS.1770 K-weighting coefficients.
    y = np.empty_like(x)
    z1 = z2 = 0.0
    for i, v in enumerate(x):
        o = b[0] * v + z1
        z1 = b[1] * v - a[1] * o + z2
        z2 = b[2] * v - a[2] * o
        y[i] = o
    return y


def lufs(x):
    # ITU-R BS.1770-4 coefficients at 48 kHz, mono channel weight 1.
    y = biquad(x, [1.53512485958697, -2.69169618940638, 1.19839281085285], [1, -1.69065929318241, .73248077421585])
    y = biquad(y, [1, -2, 1], [1, -1.99004745483398, .99007225036621])
    block, hop = int(.4 * FS), int(.1 * FS)
    z = np.cumsum(np.r_[0, y * y])
    energy = (z[block::hop] - z[:-block:hop]) / block
    preliminary = -0.691 + 10 * np.log10(np.maximum(energy, 1e-20))
    absolute = energy[preliminary > -70]
    if len(absolute) == 0:
        return -np.inf
    relative_threshold = -0.691 + 10 * np.log10(absolute.mean()) - 10
    gated = energy[(preliminary > -70) & (preliminary > relative_threshold)]
    return float(-0.691 + 10 * np.log10(gated.mean()))


def lag_estimate(ref, x):
    # Positive lag means x begins later than ref. A wide waveform search is
    # important: repetitive material can give a false envelope-correlation peak.
    dec = 8
    base, duration, reach = 10 * FS, 8 * FS, 2 * FS
    aa = ref[base:base + duration:dec]
    bb = x[base - reach:base + duration + reach:dec]
    n = 1 << (len(aa) + len(bb) - 2).bit_length()
    corr = np.fft.irfft(np.fft.rfft(bb, n) * np.conj(np.fft.rfft(aa, n)), n)
    shift = int(np.argmax(np.abs(corr[:len(bb) - len(aa) + 1])))
    center = shift * dec - reach
    # Refine around the downsampled solution to one original 48 kHz sample.
    aa = ref[base:base + duration]
    lo = base + center - dec
    bb = x[lo:lo + duration + 2 * dec]
    n = 1 << (len(aa) + len(bb) - 2).bit_length()
    corr = np.fft.irfft(np.fft.rfft(bb, n) * np.conj(np.fft.rfft(aa, n)), n)
    return center + int(np.argmax(np.abs(corr[:2 * dec + 1]))) - dec


def sync_diagnostics(ref, x):
    """Independent 5 s windows expose clock drift or an edited/discontinuous take."""
    dec = 8
    a, b = ref[::dec], x[::dec]
    sr = FS // dec
    results = []
    for sec in [5, 20, 40, 60]:
        aa = a[sec * sr:(sec + 5) * sr]
        bb = b[(sec - 1) * sr:(sec + 6) * sr]
        n = 1 << (len(aa) + len(bb) - 2).bit_length()
        c = np.fft.irfft(np.fft.rfft(bb, n) * np.conj(np.fft.rfft(aa, n)), n)
        idx = int(np.argmax(np.abs(c[:len(bb) - len(aa) + 1])))
        segment = bb[idx:idx + len(aa)]
        results.append(((idx - sr) / sr * 1000, float(np.corrcoef(aa, segment)[0, 1])))
    return results


def spectrum(x, nfft=8192, hop=4096):
    win = np.hanning(nfft)
    frames = np.lib.stride_tricks.sliding_window_view(x, nfft)[::hop]
    power = np.abs(np.fft.rfft(frames * win, axis=1)) ** 2
    frame_level = np.mean(frames ** 2, axis=1)
    # Omit near-silent intervals to avoid the room/electronic noise floor.
    keep = frame_level > max(np.quantile(frame_level, .3), 1e-9)
    return np.median(power[keep], axis=0), power, keep, np.fft.rfftfreq(nfft, 1 / FS)


def content_coverage(power, active, freq, centers):
    """Heuristic programme support, not a statistical FR confidence interval.

    It combines active-vs-quiet contrast, prevalence across active windows, and
    relative programme energy. This prevents a quiet/noise-dominated top octave
    from looking as trustworthy as a strongly excited mid band.
    """
    live = np.median(power[active], axis=0)
    quiet = np.median(power[~active], axis=0)
    contrast = 10 * np.log10(np.maximum(live, 1e-20) / np.maximum(quiet, 1e-20))
    prevalence = np.mean(power[active] > 4 * quiet, axis=0)
    energy_db = 10 * np.log10(np.maximum(live, 1e-20))
    contrast_s = np.array(smooth_log(freq, contrast, centers))
    prevalence_s = np.array(smooth_log(freq, prevalence, centers))
    energy_s = np.array(smooth_log(freq, energy_db, centers))
    peak = np.percentile(energy_s, 95)
    snr_score = np.clip((contrast_s - 3) / 15, 0, 1)
    energy_score = np.clip((energy_s - peak + 35) / 30, 0, 1)
    prevalence_score = np.clip((prevalence_s - .2) / .6, 0, 1)
    return np.sqrt(snr_score * energy_score) * (.55 + .45 * prevalence_score)


def smooth_log(freq, values, centers):
    out = []
    for f in centers:
        use = (freq >= f * 2 ** (-1 / 12)) & (freq <= f * 2 ** (1 / 12))
        out.append(float(np.median(values[use])) if use.any() else None)
    return out


def main():
    raw = {name: read(name) for name in NAMES}
    ref = raw[REF]
    lags = {REF: 0}
    diagnostics = {name: sync_diagnostics(ref, raw[name]) for name in NAMES}
    for name in NAMES:
        if name != REF:
            lags[name] = lag_estimate(ref, raw[name])
        print('lag', name, lags[name], f'{lags[name]/FS:.5f}s', flush=True)
    start = max(0, *[-lag for lag in lags.values()])
    end = min(len(raw[name]) - lags[name] for name in NAMES)
    common = int(end - start)
    assert common > 60 * FS
    polarity = {name: -1 if np.median([v[1] for v in diagnostics[name]]) < 0 else 1 for name in NAMES}
    aligned = {name: polarity[name] * raw[name][start + lags[name]:start + lags[name] + common] for name in NAMES}
    levels = {name: lufs(x) for name, x in aligned.items()}
    # Fixed common segment for all plots and A/B playback. Peak-safe gain
    # reduction is shared across all files and thus preserves matched LUFS.
    gains = {name: 10 ** ((TARGET - levels[name]) / 20) for name in NAMES}
    global_peak = max(float(np.max(np.abs(aligned[name] * gains[name]))) for name in NAMES)
    safety = min(1.0, .98 / global_peak)
    played = {name: aligned[name] * gains[name] * safety for name in NAMES}
    centers = np.geomspace(40, 18000, 120)
    spectra = {}
    coverages = {}
    for name, x in played.items():
        spec, power, active, freq = spectrum(x)
        spectra[name] = spec
        coverages[name] = content_coverage(power, active, freq, centers)
        with wave.open(str(OUT / f'{name}.wav'), 'wb') as f:
            f.setnchannels(1); f.setsampwidth(2); f.setframerate(FS)
            f.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())
    # Normalize ratios around the core speech/music mid-band. This is not
    # absolute microphone sensitivity or a calibrated transfer function.
    rows = []
    refspec = spectra[REF]
    for name in NAMES:
        ratio = 10 * np.log10(np.maximum(spectra[name], 1e-20) / np.maximum(refspec, 1e-20))
        ratio -= np.median(ratio[(freq >= 500) & (freq <= 2000)])
        relative = smooth_log(freq, ratio, centers)
        bands = {}
        for label, low, high in [('bass', 80, 250), ('lowMid', 250, 500), ('presence', 2000, 5000), ('air', 8000, 16000)]:
            bands[label] = round(float(np.median(ratio[(freq >= low) & (freq < high)])), 1)
        # Spectral variability in the high range flags programme/noise limits.
        mid = (freq >= 500) & (freq < 2000)
        high = (freq >= 8000) & (freq < 16000)
        high_floor = 10 * np.log10(np.maximum(spectra[name][high].mean(), 1e-20) / np.maximum(spectra[name][mid].mean(), 1e-20))
        sync_spread = max(v[0] for v in diagnostics[name]) - min(v[0] for v in diagnostics[name])
        rows.append(dict(id=name, label=name, originalLufs=round(levels[name], 2), gainDb=round(20*np.log10(gains[name]),2), lagMs=round(lags[name]/FS*1000,2), syncSpreadMs=round(sync_spread,1), polarityInverted=polarity[name]<0, peakDbfs=round(20*np.log10(np.max(np.abs(played[name]))),1), bands=bands, highToMidDb=round(float(high_floor),1), curve=[round(v,2) if v is not None else None for v in relative], coverage=[round(float(v),3) for v in coverages[name]], audio=f'data/{name}.wav'))
        print(name, 'LUFS', levels[name], 'bands', bands, flush=True)
    overall_coverage = np.median(np.stack(list(coverages.values())), axis=0)
    payload = dict(reference=REF, targetLufs=TARGET+20*np.log10(safety), durationSec=round(common/FS,3), sampleRate=FS, frequencies=[round(float(v),1) for v in centers], coverage=[round(float(v),3) for v in overall_coverage], microphones=rows, note='Programme-dependent relative spectral estimate, not calibrated microphone FR. Coverage is a heuristic based on active-vs-quiet contrast, prevalence, and relative energy—not a calibrated confidence interval. Curves normalized to 0 dB median at 500–2000 Hz. Same-source placement, room, polar pattern, preamp, device processing and noise remain confounds. em-usb has a variable lag and is only approximately aligned.')
    (OUT/'analysis.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print('saved', OUT, 'duration', common/FS, 'safety dB', 20*np.log10(safety))


if __name__ == '__main__':
    main()
