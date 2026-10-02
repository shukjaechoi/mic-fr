"""Compare five simultaneous microphone/device recordings from 2026-10-02.

The ``speaker`` takes provide the broad-band reference measurement.  The
``guitar`` takes test how closely programme material reproduces that estimate.
Input files are canonical 48 kHz mono PCM16 WAV files made from the originals;
set INPUT_DIR to their directory before running if needed.

Only NumPy is required.  Results are relative to the KM184 and must not be
interpreted as calibrated free-field microphone responses.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import struct
import subprocess
import wave
from pathlib import Path

import numpy as np


ROOT = Path(__file__).parent
SOURCE = Path(os.environ.get("SOURCE_DIR", "/Users/sjchoi/dev/rec/261002-iphone,h1,km184,s21,d10"))
INPUT = Path(os.environ.get("INPUT_DIR", "/private/tmp/mic-fr-261002"))
OUT = ROOT / "docs" / "experiments" / "20261002" / "data"
FS = 48_000
TARGET_LUFS = -20.0
EXCERPT_SECONDS = 90
NFFT = 8192
IDS = ["iphone", "h1", "km184", "s21", "d10"]
LABELS = {
    "iphone": "iPhone",
    "h1": "Zoom H1",
    "km184": "Neumann KM184",
    "s21": "Galaxy S21",
    "d10": "Sony PCM-D10",
}
REF = "km184"
ORIGINALS = {
    ("iphone", "speaker"): "iphone speaker.m4a",
    ("h1", "speaker"): "h1 speaker.WAV",
    ("km184", "speaker"): "neumann km184 speaker.wav",
    ("s21", "speaker"): "s21 speaker.wav",
    ("d10", "speaker"): "sony pcm d10 speaker.wav",
    ("iphone", "guitar"): "iphone guitar.m4a",
    ("h1", "guitar"): "h1 guitar.WAV",
    ("km184", "guitar"): "neumann km184 guitar.wav",
    ("s21", "guitar"): "s21 guitar.wav",
    ("d10", "guitar"): "sony pcm d10 guitar.wav",
}


def prepare_inputs() -> None:
    """Decode/downmix originals to canonical WAVs when the cache is absent."""
    missing = [(name, kind) for name in IDS for kind in ("speaker", "guitar")
               if not (INPUT / f"{name}-{kind}.wav").exists()]
    if not missing:
        return
    ffmpeg = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to decode the source files; set FFMPEG=/path/to/ffmpeg")
    INPUT.mkdir(parents=True, exist_ok=True)
    for name, kind in missing:
        source = SOURCE / ORIGINALS[(name, kind)]
        destination = INPUT / f"{name}-{kind}.wav"
        print("decode", source.name, "->", destination, flush=True)
        subprocess.run([
            ffmpeg, "-hide_banner", "-loglevel", "warning", "-y", "-i", str(source),
            "-ac", "1", "-ar", str(FS), "-c:a", "pcm_s16le", str(destination),
        ], check=True)


def pcm16_mmap(path: Path) -> np.memmap:
    """Memory-map the data chunk of a mono PCM16 RIFF/WAVE file."""
    with path.open("rb") as f:
        if f.read(4) != b"RIFF":
            raise ValueError(f"Not RIFF: {path}")
        f.seek(8)
        if f.read(4) != b"WAVE":
            raise ValueError(f"Not WAVE: {path}")
        channels = bits = rate = None
        data_offset = data_size = None
        while True:
            header = f.read(8)
            if len(header) < 8:
                break
            chunk_id, size = struct.unpack("<4sI", header)
            start = f.tell()
            if chunk_id == b"fmt ":
                fmt, channels, rate, _, _, bits = struct.unpack("<HHIIHH", f.read(16))
                if fmt != 1:
                    raise ValueError(f"Expected PCM WAV: {path}")
            elif chunk_id == b"data":
                data_offset, data_size = start, size
                break
            f.seek(start + size + (size & 1))
    if (channels, bits, rate) != (1, 16, FS) or data_offset is None:
        raise ValueError(f"Expected mono PCM16 {FS} Hz: {path}")
    return np.memmap(path, dtype="<i2", mode="r", offset=data_offset, shape=(data_size // 2,))


def rms_envelope(x: np.ndarray, rate: int = 20) -> np.ndarray:
    step = FS // rate
    count = len(x) // step
    # Read in modest blocks so a long mmap never becomes one huge float array.
    result = np.empty(count, np.float64)
    block = 2000
    for first in range(0, count, block):
        last = min(count, first + block)
        samples = np.asarray(x[first * step:last * step], np.float64).reshape(-1, step)
        result[first:last] = np.sqrt(np.mean(samples * samples, axis=1))
    return np.log1p(result / 32.0)


def fft_corr(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full cross-correlation: result lag k maps b index to a index + k."""
    n = 1 << (len(a) + len(b) - 2).bit_length()
    result = np.fft.irfft(np.fft.rfft(a, n) * np.fft.rfft(b[::-1], n), n)
    return result[: len(a) + len(b) - 1]


def global_lag(ref: np.ndarray, target: np.ndarray) -> int:
    a = (target - np.mean(target)) / (np.std(target) + 1e-12)
    b = (ref - np.mean(ref)) / (np.std(ref) + 1e-12)
    corr = fft_corr(a, b)
    lags = np.arange(-(len(ref) - 1), len(target))
    overlap = np.minimum(len(ref), len(target) - lags) - np.maximum(0, -lags)
    corr[overlap < 0.45 * min(len(ref), len(target))] = 0
    return int(lags[np.argmax(np.abs(corr))])


def local_lag(ref: np.ndarray, target: np.ndarray, ref_start: int, predicted: int,
              window: int, reach: int) -> tuple[int, float]:
    aa = ref[ref_start:ref_start + window].astype(np.float64)
    lo = max(0, predicted - reach)
    hi = min(len(target), predicted + window + reach)
    bb = target[lo:hi].astype(np.float64)
    if len(aa) < window or len(bb) < window:
        return predicted - ref_start, 0.0
    aa -= aa.mean()
    aa /= np.linalg.norm(aa) + 1e-12
    corr = fft_corr(bb, aa)
    # Valid placements of aa inside bb have full overlap.
    valid = corr[window - 1:len(bb)]
    starts = np.arange(len(valid))
    sums = np.r_[0.0, np.cumsum(bb)]
    sums2 = np.r_[0.0, np.cumsum(bb * bb)]
    energy = sums2[starts + window] - sums2[starts] - (
        sums[starts + window] - sums[starts]
    ) ** 2 / window
    score = valid / np.sqrt(np.maximum(energy, 1e-20))
    best = int(np.argmax(np.abs(score)))
    return lo + best - ref_start, float(score[best])


def estimate_mapping(ref_env: np.ndarray, target_env: np.ndarray, rate: int = 20) -> dict:
    """Fit target_sample = slope * reference_sample + intercept."""
    initial = global_lag(ref_env, target_env)
    ref_lo = max(0, -initial)
    ref_hi = min(len(ref_env), len(target_env) - initial)
    window = 45 * rate
    margin = window // 2 + 5 * rate
    anchors = np.linspace(ref_lo + margin, ref_hi - margin, 9).astype(int)
    offsets, scores, positions = [], [], []
    for center in anchors:
        r0 = center - window // 2
        lag, score = local_lag(ref_env, target_env, r0, r0 + initial, window, 5 * rate)
        if abs(score) > 0.08:
            positions.append(center)
            offsets.append(lag)
            scores.append(score)
    if len(positions) >= 3:
        slope_delta, intercept = np.polyfit(positions, offsets, 1)
        slope = 1.0 + float(slope_delta)
    else:
        slope, intercept = 1.0, float(initial)
    predicted = slope * np.asarray(positions) + intercept
    residual = (np.asarray(positions) + np.asarray(offsets) - predicted) / rate * 1000
    return {
        "slope": slope,
        "interceptSamples": float(intercept / rate * FS),
        "initialLagSec": initial / rate,
        "driftPpm": (slope - 1.0) * 1e6,
        "residualMs": float(np.sqrt(np.mean(residual * residual))) if len(residual) else None,
        "anchorCorrelation": float(np.median(np.abs(scores))) if scores else 0.0,
    }


def refine_mapping_audio(ref: np.ndarray, target: np.ndarray, coarse: dict) -> dict:
    """Resolve repetitive-envelope ambiguity with waveform correlation."""
    initial = int(round(coarse["initialLagSec"] * FS))
    ref_lo = max(0, -initial)
    ref_hi = min(len(ref), len(target) - initial)
    window, reach, dec = 12 * FS, int(1.5 * FS), 8
    anchors = np.linspace(ref_lo + window, ref_hi - window, 9).astype(np.int64)
    x_positions, y_positions, scores = [], [], []
    for center in anchors:
        r0 = int(center - window // 2)
        predicted = r0 + initial
        lo = max(0, predicted - reach)
        hi = min(len(target), predicted + window + reach)
        aa = np.asarray(ref[r0:r0 + window:dec], np.float64)
        bb = np.asarray(target[lo:hi:dec], np.float64)
        aa -= aa.mean()
        aa /= np.linalg.norm(aa) + 1e-12
        bb -= bb.mean()
        corr = fft_corr(bb, aa)
        valid = corr[len(aa) - 1:len(bb)]
        starts = np.arange(len(valid))
        sums2 = np.r_[0.0, np.cumsum(bb * bb)]
        energy = sums2[starts + len(aa)] - sums2[starts]
        score = valid / np.sqrt(np.maximum(energy, 1e-20))
        best = int(np.argmax(np.abs(score)))
        target_start = lo + best * dec
        if abs(score[best]) >= .08:
            x_positions.append(r0)
            y_positions.append(target_start)
            scores.append(float(score[best]))
    if len(x_positions) < 3:
        return coarse
    slope, intercept = np.polyfit(x_positions, y_positions, 1)
    predicted = slope * np.asarray(x_positions) + intercept
    residual = (np.asarray(y_positions) - predicted) / FS * 1000
    # A bad fit usually means the test signal contains repeated passages. In
    # that case the envelope mapping is safer than a false waveform peak.
    if np.sqrt(np.mean(residual * residual)) > 80 or not (.998 < slope < 1.002):
        return coarse
    return {
        **coarse,
        "slope": float(slope),
        "interceptSamples": float(intercept),
        "driftPpm": float((slope - 1) * 1e6),
        "residualMs": float(np.sqrt(np.mean(residual * residual))),
        "anchorCorrelation": float(np.median(np.abs(scores))),
    }


def common_reference_range(maps: dict, audio: dict) -> tuple[int, int]:
    lo, hi = 0.0, float(len(audio[REF]) - 1)
    for name in IDS:
        slope = maps[name]["slope"]
        intercept = maps[name]["interceptSamples"]
        lo = max(lo, -intercept / slope)
        hi = min(hi, (len(audio[name]) - 1 - intercept) / slope)
    return int(math.ceil(lo)), int(math.floor(hi))


def resample_from_reference(x: np.ndarray, mapping: dict, ref_positions: np.ndarray) -> np.ndarray:
    positions = mapping["slope"] * ref_positions + mapping["interceptSamples"]
    left = int(np.floor(positions.min()))
    right = int(np.ceil(positions.max())) + 1
    chunk = np.asarray(x[left:right], np.float64) / 32768.0
    return np.interp(positions, np.arange(left, right), chunk)


def biquad(x: np.ndarray, b: list[float], a: list[float]) -> np.ndarray:
    y = np.empty_like(x)
    z1 = z2 = 0.0
    for i, value in enumerate(x):
        out = b[0] * value + z1
        z1 = b[1] * value - a[1] * out + z2
        z2 = b[2] * value - a[2] * out
        y[i] = out
    return y


def lufs(x: np.ndarray) -> float:
    y = biquad(x, [1.53512485958697, -2.69169618940638, 1.19839281085285],
               [1, -1.69065929318241, .73248077421585])
    y = biquad(y, [1, -2, 1], [1, -1.99004745483398, .99007225036621])
    block, hop = int(.4 * FS), int(.1 * FS)
    z = np.cumsum(np.r_[0, y * y])
    energy = (z[block::hop] - z[:-block:hop]) / block
    levels = -0.691 + 10 * np.log10(np.maximum(energy, 1e-20))
    absolute = energy[levels > -70]
    relative = -0.691 + 10 * np.log10(absolute.mean()) - 10
    gated = energy[(levels > -70) & (levels > relative)]
    return float(-0.691 + 10 * np.log10(gated.mean()))


def smooth_log(freq: np.ndarray, values: np.ndarray, centers: np.ndarray,
               reducer=np.median) -> np.ndarray:
    result = []
    for center in centers:
        use = (freq >= center * 2 ** (-1 / 12)) & (freq <= center * 2 ** (1 / 12))
        result.append(float(reducer(values[use])) if np.any(use) else np.nan)
    return np.asarray(result)


def frame_starts(ref_audio: np.ndarray, lo: int, hi: int, maximum: int = 1000) -> np.ndarray:
    candidates = np.arange(lo, hi - NFFT, FS // 2, dtype=np.int64)
    rms = np.empty(len(candidates))
    for i, start in enumerate(candidates):
        frame = np.asarray(ref_audio[start:start + NFFT], np.float64)
        rms[i] = np.mean(frame * frame)
    keep = rms >= np.quantile(rms, .20)
    active = candidates[keep]
    if len(active) > maximum:
        active = active[np.linspace(0, len(active) - 1, maximum).astype(int)]
    quiet = candidates[rms <= np.quantile(rms, .12)]
    if len(quiet) > 250:
        quiet = quiet[np.linspace(0, len(quiet) - 1, 250).astype(int)]
    return active, quiet


def powers_for_frames(x: np.ndarray, mapping: dict, starts: np.ndarray) -> np.ndarray:
    win = np.hanning(NFFT)
    power = np.empty((len(starts), NFFT // 2 + 1), np.float64)
    local = np.arange(NFFT, dtype=np.float64)
    for i, start in enumerate(starts):
        ref_positions = start + local
        frame = resample_from_reference(x, mapping, ref_positions)
        power[i] = np.abs(np.fft.rfft(frame * win)) ** 2
    return power


def analyse_spectra(kind: str, audio: dict, maps: dict, centers: np.ndarray) -> dict:
    lo, hi = common_reference_range(maps, audio)
    active, quiet = frame_starts(audio[REF], lo, hi)
    freq = np.fft.rfftfreq(NFFT, 1 / FS)
    powers = {}
    for name in IDS:
        powers[name] = powers_for_frames(audio[name], maps[name], active)
    ref_mean = np.mean(powers[REF], axis=0)
    ref_quiet = powers_for_frames(audio[REF], maps[REF], quiet)
    quiet_mean = np.median(ref_quiet, axis=0)
    energy_db = 10 * np.log10(np.maximum(ref_mean, 1e-30))
    contrast_db = 10 * np.log10(np.maximum(ref_mean, 1e-30) / np.maximum(quiet_mean, 1e-30))
    energy_s = smooth_log(freq, energy_db, centers)
    contrast_s = smooth_log(freq, contrast_db, centers)
    peak = np.percentile(energy_s[np.isfinite(energy_s)], 95)
    # Energy support is deliberately strict in the extremes.  It describes
    # stimulus support, not a formal confidence interval.
    energy_score = np.clip((energy_s - peak + 42) / 32, 0, 1)
    contrast_score = np.clip((contrast_s - 4) / 22, 0, 1)
    coverage = np.sqrt(energy_score * contrast_score)
    curves = {}
    spread = {}
    for name in IDS:
        mean_power = np.mean(powers[name], axis=0)
        ratio = 10 * np.log10(np.maximum(mean_power, 1e-30) / np.maximum(ref_mean, 1e-30))
        mid = (freq >= 500) & (freq <= 2000)
        ratio -= np.median(ratio[mid])
        curves[name] = smooth_log(freq, ratio, centers)
        per_frame = 10 * np.log10(np.maximum(powers[name], 1e-30) / np.maximum(powers[REF], 1e-30))
        frame_spread = np.median(np.abs(per_frame - np.median(per_frame, axis=0)), axis=0) * 1.4826
        spread[name] = smooth_log(freq, frame_spread, centers)
    return {
        "kind": kind,
        "range": (lo, hi),
        "frames": len(active),
        "coverage": coverage,
        "curves": curves,
        "spread": spread,
    }


def choose_excerpt(ref_env: np.ndarray, common: tuple[int, int], rate: int = 20) -> int:
    lo, hi = [int(v / FS * rate) for v in common]
    length = EXCERPT_SECONDS * rate
    # Prefer the middle-to-late portion but avoid tails and pick a musically
    # active window rather than a fixed timestamp.
    search_lo = lo + int((hi - lo) * .25)
    search_hi = lo + int((hi - lo) * .85) - length
    values = np.exp(ref_env) - 1
    rolling = np.convolve(values, np.ones(length), mode="valid")
    start = search_lo + int(np.argmax(rolling[search_lo:search_hi + 1]))
    return int(start / rate * FS)


def write_wave(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(FS)
        f.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def waveform_envelope(x: np.ndarray, bins: int = 1200) -> list[float]:
    edges = np.linspace(0, len(x), bins + 1, dtype=int)
    result = []
    for a, b in zip(edges[:-1], edges[1:]):
        part = x[a:b:max(1, (b - a) // 256)]
        result.append(round(float(np.sqrt(np.mean(part * part))), 6))
    return result


def hz_label(value: float) -> str:
    return f"{value / 1000:.1f} kHz" if value >= 1000 else f"{value:.0f} Hz"


def supported_band(coverage: np.ndarray, centers: np.ndarray, threshold: float = .4) -> list[float]:
    good = np.flatnonzero(coverage >= threshold)
    return [round(float(centers[good[0]]), 1), round(float(centers[good[-1]]), 1)] if len(good) else []


def main() -> None:
    prepare_inputs()
    OUT.mkdir(parents=True, exist_ok=True)
    all_audio, all_env, all_maps = {}, {}, {}
    for kind in ("speaker", "guitar"):
        audio = {name: pcm16_mmap(INPUT / f"{name}-{kind}.wav") for name in IDS}
        env = {name: rms_envelope(x) for name, x in audio.items()}
        maps = {REF: {"slope": 1.0, "interceptSamples": 0.0, "initialLagSec": 0.0,
                      "driftPpm": 0.0, "residualMs": 0.0, "anchorCorrelation": 1.0}}
        for name in IDS:
            if name != REF:
                maps[name] = estimate_mapping(env[REF], env[name])
                maps[name] = refine_mapping_audio(audio[REF], audio[name], maps[name])
                if abs(maps[name]["driftPpm"]) > 300 or (maps[name]["residualMs"] or 0) > 80:
                    maps[name] = {
                        **maps[name],
                        "slope": 1.0,
                        "interceptSamples": maps[name]["initialLagSec"] * FS,
                        "driftPpm": 0.0,
                        "residualMs": None,
                        "mappingMethod": "offset-only (repetitive stimulus)",
                    }
                else:
                    maps[name]["mappingMethod"] = "offset + clock drift"
            else:
                maps[name]["mappingMethod"] = "reference"
            print(kind, name, maps[name], flush=True)
        all_audio[kind], all_env[kind], all_maps[kind] = audio, env, maps

    centers = np.geomspace(40, 18_000, 120)
    results = {
        kind: analyse_spectra(kind, all_audio[kind], all_maps[kind], centers)
        for kind in ("speaker", "guitar")
    }

    guitar_common = common_reference_range(all_maps["guitar"], all_audio["guitar"])
    excerpt_start = choose_excerpt(all_env["guitar"][REF], guitar_common)
    ref_positions = excerpt_start + np.arange(EXCERPT_SECONDS * FS, dtype=np.float64)
    excerpt = {
        name: resample_from_reference(all_audio["guitar"][name], all_maps["guitar"][name], ref_positions)
        for name in IDS
    }
    # Correct polarity only for seamless switching; the spectral analysis uses
    # power and is polarity-independent.
    for name in IDS:
        if name != REF and np.corrcoef(excerpt[REF][:FS * 20], excerpt[name][:FS * 20])[0, 1] < 0:
            excerpt[name] *= -1
    original_lufs = {name: lufs(x) for name, x in excerpt.items()}
    gains = {name: 10 ** ((TARGET_LUFS - level) / 20) for name, level in original_lufs.items()}
    peak = max(float(np.max(np.abs(excerpt[name] * gains[name]))) for name in IDS)
    safety = min(1.0, .98 / peak)
    played = {name: excerpt[name] * gains[name] * safety for name in IDS}
    for name in IDS:
        write_wave(OUT / f"{name}-guitar.wav", played[name])
    (OUT / "waveform.json").write_text(json.dumps(waveform_envelope(played[REF])))

    comparison_rows = []
    microphones = []
    both_coverage = np.minimum(results["speaker"]["coverage"], results["guitar"]["coverage"])
    use = both_coverage >= .35
    for name in IDS:
        speaker_curve = results["speaker"]["curves"][name]
        guitar_curve = results["guitar"]["curves"][name]
        diff = guitar_curve - speaker_curve
        if name == REF:
            rmse, corr = 0.0, 1.0
        else:
            weights = both_coverage[use]
            rmse = float(np.sqrt(np.average(diff[use] ** 2, weights=weights)))
            corr = float(np.corrcoef(speaker_curve[use], guitar_curve[use])[0, 1])
        comparison_rows.append({"id": name, "rmseDb": round(rmse, 2), "correlation": round(corr, 3)})
        microphones.append({
            "id": name,
            "label": LABELS[name],
            "audio": f"data/{name}-guitar.wav",
            "originalLufs": round(original_lufs[name], 2),
            "gainDb": round(20 * np.log10(gains[name]), 2),
            "speakerCurve": np.round(speaker_curve, 2).tolist(),
            "guitarCurve": np.round(guitar_curve, 2).tolist(),
            "speakerSpread": np.round(results["speaker"]["spread"][name], 2).tolist(),
            "guitarSpread": np.round(results["guitar"]["spread"][name], 2).tolist(),
            "speakerSync": all_maps["speaker"][name],
            "guitarSync": all_maps["guitar"][name],
        })
        print(name, "LUFS", original_lufs[name], "speaker/guitar RMSE", rmse, "corr", corr, flush=True)

    payload = {
        "title": "2026-10-02 · speaker / guitar 비교",
        "reference": REF,
        "sampleRate": FS,
        "targetLufs": round(TARGET_LUFS + 20 * np.log10(safety), 2),
        "durationSec": EXCERPT_SECONDS,
        "excerptSourceStartSec": round(excerpt_start / FS, 2),
        "frequencies": np.round(centers, 1).tolist(),
        "speakerCoverage": np.round(results["speaker"]["coverage"], 3).tolist(),
        "guitarCoverage": np.round(results["guitar"]["coverage"], 3).tolist(),
        "speakerSupportedBand": supported_band(results["speaker"]["coverage"], centers),
        "guitarSupportedBand": supported_band(results["guitar"]["coverage"], centers),
        "comparisonBand": supported_band(both_coverage, centers, .35),
        "speakerFrames": results["speaker"]["frames"],
        "guitarFrames": results["guitar"]["frames"],
        "comparison": comparison_rows,
        "microphones": microphones,
        "method": (
            "각 세트는 RMS 엔벌로프로 시작 오프셋과 선형 클록 드리프트를 추정한 뒤 KM184 시간축에 재표본화했다. "
            "곡선은 동일 시각 STFT 전력비를 1/6옥타브로 평활하고 500–2000 Hz 중앙값을 0 dB로 맞춘 상대값이다. "
            "coverage는 KM184 기준 신호 에너지와 무음 대비를 합친 휴리스틱이며 교정된 신뢰구간은 아니다."
        ),
        "limits": (
            "speaker 결과도 방·스피커·배치의 영향을 포함하며 절대 FR이 아니다. guitar 결과는 연주에 충분히 포함된 대역에서만 "
            "speaker 추정과 비교할 수 있다. 특히 저역과 고역에서 coverage가 낮으면 곡선 차이를 마이크 성능으로 단정할 수 없다."
        ),
    }
    (OUT / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print("saved", OUT, flush=True)


if __name__ == "__main__":
    main()
