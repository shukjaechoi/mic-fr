"""Add the iPhone → NAM A2 Full-500 result to the mic-fr listening page.

The NAM model is run on the original iPhone timeline.  This script then finds
the existing page iPhone track inside that timeline, so its export has exactly
the same time origin and duration as every mic-fr playback track.
"""
from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np
import soundfile as sf

from analyze import FS, content_coverage, lag_estimate, lufs, smooth_log, spectrum


DATA = Path(__file__).parent / "docs" / "data"
DEFAULT_INPUT = Path(__file__).parents[2] / "runs" / "comparison" / "iphone_nam_a2_full_500_timeline.wav"
ID = "iphone-nam-a2-full-500"


def read_wav(path: Path) -> np.ndarray:
    # soundfile handles the PCM16 page assets as well as PCM24/float model
    # exports, which avoids silently misreading a 24-bit output as float32.
    frames, sample_rate = sf.read(path, always_2d=True)
    if sample_rate != FS:
        raise ValueError(f"Expected 48 kHz: {path}")
    return frames.mean(axis=1).astype(np.float64)


def write_wav(path: Path, x: np.ndarray) -> None:
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(FS)
        wav.writeframes((np.clip(x, -1, 1) * 32767).astype("<i2").tobytes())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--id", default=ID)
    parser.add_argument("--label", default="iPhone - NAM A2 Full-500")
    parser.add_argument("--note", default=("iPhone - NAM A2 Full-500 is the selected epoch-470 checkpoint from a 500-epoch "
                                             "paired iPhone→KM184 run. The browser export is cropped to the mic-fr aligned iPhone "
                                             "timeline and LUFS-matched with the other tracks. It is a same-recording experiment, "
                                             "not a general microphone emulation claim."))
    args = parser.parse_args()

    payload = json.loads((DATA / "analysis.json").read_text())
    payload["microphones"] = [m for m in payload["microphones"] if m["id"] != args.id]
    iphone = next(m for m in payload["microphones"] if m["id"] == "iphone")
    page_phone = read_wav(DATA / "iphone.wav")
    raw_model = read_wav(args.input)

    # Same convention as analyze.py: positive lag means the raw model timeline
    # begins later than the page's aligned iPhone reference timeline.
    lag = lag_estimate(page_phone, raw_model)
    start = max(0, lag)
    model = raw_model[start:start + len(page_phone)]
    if len(model) != len(page_phone):
        raise RuntimeError("Model export does not cover the page comparison interval")

    # The listener switches tracks at a shared target LUFS, as it does for the
    # original microphones and the FIR/IIR derived tracks.
    model *= 10 ** ((float(payload["targetLufs"]) - lufs(model)) / 20)
    output = DATA / f"{args.id}.wav"
    write_wav(output, model)
    model = read_wav(output)  # analyze the exact browser deliverable

    ref = read_wav(DATA / "km184.wav")
    spec, power, active, freq = spectrum(model)
    ref_spec, _, _, _ = spectrum(ref)
    ratio = 10 * np.log10(np.maximum(spec, 1e-20) / np.maximum(ref_spec, 1e-20))
    ratio -= np.median(ratio[(freq >= 500) & (freq <= 2000)])
    bands = {key: round(float(np.median(ratio[(freq >= lo) & (freq < hi)])), 1)
             for key, lo, hi in [("bass", 80, 250), ("lowMid", 250, 500),
                                 ("presence", 2000, 5000), ("air", 8000, 16000)]}
    centers = np.asarray(payload["frequencies"])
    row = dict(
        id=args.id, label=args.label, derivedFrom="iphone",
        originalLufs=iphone["originalLufs"],
        gainDb=round(float(payload["targetLufs"]) - lufs(raw_model[start:start + len(page_phone)]), 2),
        processedLufs=round(lufs(model), 2), lagMs=iphone["lagMs"],
        syncSpreadMs=iphone["syncSpreadMs"], polarityInverted=False,
        peakDbfs=round(20 * np.log10(np.max(np.abs(model))), 1), bands=bands,
        highToMidDb=round(float(10 * np.log10(np.maximum(spec[(freq >= 8000) & (freq < 16000)].mean(), 1e-20) /
                                                     np.maximum(spec[(freq >= 500) & (freq < 2000)].mean(), 1e-20))), 1),
        curve=[round(v, 2) if v is not None else None for v in smooth_log(freq, ratio, centers)],
        coverage=[round(float(v), 3) for v in content_coverage(power, active, freq, centers)],
        audio=f"data/{args.id}.wav")
    index = next(i for i, m in enumerate(payload["microphones"]) if m["id"] == "iphone")
    payload["microphones"].insert(index + 1, row)
    payload["namNote"] = args.note
    (DATA / "analysis.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(f"saved {output}; lag {lag} samples ({lag / FS * 1000:.2f} ms); bands {bands}")


if __name__ == "__main__":
    main()
