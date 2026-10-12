# Reference-assisted magnitude diagnostic — 2026-10-12

Purpose: distinguish shared target-curve limitations from time-varying magnitude limitations without assuming remaining error is phase, reverb, or a need for neural processing.

Frozen comparisons: October KM184 paired with iPhone/S21; arpeggio 100–105s, complex chords 1704–1709s, percussion 1792–1797s. Baseline = latest local-resolution 1/12 tonal SOS, identically applied with real context. Existing clip level anchor frozen for all variants. These are already explored examples and fitting/evaluation use the same clips: diagnostic fit, no generalization claim.

Two supplements share 1/6 octave centers 600Hz–12kHz, ±6dB, 4096-sample Hann STFT/hop512 at48kHz, frequency smoothing0.6 grid steps, same raw joint signal confidence, same processing/tapers and device STFT phase. Static supplement pools the clip. Temporal supplement alone can vary in time, smooths over60ms and rate-limits to24dB/s. Missing observation shrinks toward baseline; no target magnitude copying in unsupported bins. Outside500–14kHz unity. This STFT implementation is offline and centered; no real-time latency claim.

Measures: original-mask weighted absolute/signed band error, independent short-time broad-band attack/body/decay errors, KM-selected spectral peak/flank contrast, gain extent and temporal rate, common preview peak and4x oversampled peak, baseline STFT reconstruction. Store complete gain grids and confidence. Do not change rules based on results.

Interpretation: clip-specific static benefit indicates a target selection question, not proof input-only prediction is possible. Temporal additional benefit motivates slow EQ switching or dynamic EQ; neural models only if simpler rules cannot predict required gains on other phrases. Little benefit does not diagnose phase alone. Next work is input-only predictability/other-phrase evaluation, or broader alignment/channel/phase diagnosis, depending results. September excluded.

Implementation QC: static confidence is conditional on observed frames (>=3), not diluted by silent-frame count; static spectral pooling uses original-confidence weights. Peak/flank contrast requires both the peak and flanks to clear joint10dB SNR. Percussion stage gates reuse previously recorded common event onsets. These corrections avoid unequal confidence treatment and weak-denominator comparisons.
