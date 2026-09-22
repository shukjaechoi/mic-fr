"""Create a conservative, programme-derived iPhone-to-KM184 EQ comparison.

Run after analyze.py. The filter is not a calibrated microphone transfer
function: it only targets the relative coloration in these recordings.
"""
from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np

from analyze import FS, lufs, spectrum, smooth_log, content_coverage

DATA = Path(__file__).parent / 'docs' / 'data'
ANALYSIS = DATA / 'analysis.json'
OUT = DATA / 'iphone2km184.wav'


def read_wav(path):
    with wave.open(str(path), 'rb') as wav:
        assert wav.getnchannels() == 1 and wav.getsampwidth() == 2 and wav.getframerate() == FS
        return np.frombuffer(wav.readframes(wav.getnframes()), '<i2').astype(np.float64) / 32768


def write_wav(path, x):
    with wave.open(str(path), 'wb') as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(FS)
        wav.writeframes((np.clip(x, -1, 1) * 32767).astype('<i2').tobytes())


def design_fir(frequencies, curve):
    # Curve is already 1/6-octave smoothed. Additional smoothing suppresses
    # narrow, placement-dependent notches that should not be inverse-filtered.
    kernel_x = np.arange(-9, 10)
    kernel = np.exp(-.5 * (kernel_x / 3) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(-np.asarray(curve), (9, 9), mode='edge')
    smooth_db = np.convolve(padded, kernel, mode='valid')
    n_design = 65536
    fft_freq = np.fft.rfftfreq(n_design, 1 / FS)
    lookup = np.interp(np.log(np.maximum(fft_freq, 1)), np.log(frequencies), smooth_db)
    # No high-frequency noise inversion; transition smoothly to flat gain.
    low = np.clip((fft_freq - 60) / 35, 0, 1)
    high = np.clip((5000 - fft_freq) / 1500, 0, 1)
    taper = (low * low * (3 - 2 * low)) * (high * high * (3 - 2 * high))
    target_db = np.clip(lookup, -6, 8) * taper
    response = 10 ** (target_db / 20)
    circular = np.fft.irfft(response, n_design)
    taps = 4097
    half = taps // 2
    h = np.r_[circular[-half:], circular[:half + 1]] * np.kaiser(taps, 8.6)
    h /= h.sum()  # Unity DC gain.
    return h


def fft_filter(x, h):
    n = 1 << (len(x) + len(h) - 2).bit_length()
    full = np.fft.irfft(np.fft.rfft(x, n) * np.fft.rfft(h, n), n)
    delay = len(h) // 2
    return full[delay:delay + len(x)]


def main():
    payload = json.loads(ANALYSIS.read_text())
    payload['microphones'] = [m for m in payload['microphones'] if m['id'] != 'iphone2km184']
    iphone = next(m for m in payload['microphones'] if m['id'] == 'iphone')
    centers = np.asarray(payload['frequencies'])
    x = read_wav(DATA / 'iphone.wav')
    ref = read_wav(DATA / 'km184.wav')
    assert len(x) == len(ref)
    y = fft_filter(x, design_fir(centers, iphone['curve']))
    before = lufs(y)
    target = float(payload['targetLufs'])
    gain_db = target - before
    y *= 10 ** (gain_db / 20)
    peak = float(np.max(np.abs(y)))
    if peak >= .98:
        raise RuntimeError(f'LUFS match would clip ({peak:.3f}); lower the EQ boost and retry')
    write_wav(OUT, y)
    # Analyze the quantized deliverable, not only the in-memory prediction.
    y = read_wav(OUT)
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
    row = dict(id='iphone2km184', label='iphone2km184', derivedFrom='iphone',
               originalLufs=iphone['originalLufs'], gainDb=round(gain_db, 2),
               processedLufs=round(lufs(y), 2), lagMs=iphone['lagMs'],
               syncSpreadMs=iphone['syncSpreadMs'], polarityInverted=False,
               peakDbfs=round(20 * np.log10(np.max(np.abs(y))), 1), bands=bands,
               highToMidDb=round(float(high_floor), 1),
               curve=[round(v, 2) if v is not None else None for v in smooth_log(freq, ratio, centers)],
               coverage=[round(float(v), 3) for v in coverage], audio='data/iphone2km184.wav')
    index = next(i for i, m in enumerate(payload['microphones']) if m['id'] == 'iphone')
    payload['microphones'].insert(index + 1, row)
    payload['eqNote'] = ('iphone2km184 is a programme-derived, symmetric 4097-tap FIR EQ of the aligned '
                         'iPhone take. It targets the smoothed iPhone/KM184 median-spectrum difference mainly '
                         'from 70 Hz to 4 kHz, tapers to no correction outside, and limits boost/cut to +8/-6 dB. '
                         'It is LUFS-matched again and is not a calibrated microphone emulation; noise, distortion, '
                         'DSP, phase, room and directional differences remain.')
    ANALYSIS.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print('saved', OUT, 'LUFS', row['processedLufs'], 'peak dBFS', row['peakDbfs'], 'bands', bands)


if __name__ == '__main__':
    main()
