"""Build a tiny waveform preview from the aligned KM184 listening export.

Uses only the Python standard library; run after analyze.py if the audio changes.
"""
import json
import math
import struct
import wave
from pathlib import Path

data_dir = Path(__file__).parent / 'docs' / 'data'
with wave.open(str(data_dir / 'km184.wav'), 'rb') as wav:
    assert wav.getnchannels() == 1 and wav.getsampwidth() == 2
    frames = wav.getnframes()
    pcm = wav.readframes(frames)

bins = 1200
envelope = []
for i in range(bins):
    start = i * frames // bins
    end = (i + 1) * frames // bins
    step = max(1, (end - start) // 256)
    samples = (struct.unpack_from('<h', pcm, j * 2)[0] / 32768 for j in range(start, end, step))
    squared = [sample * sample for sample in samples]
    envelope.append(round(math.sqrt(sum(squared) / len(squared)), 6))

(data_dir / 'waveform.json').write_text(json.dumps(envelope))
