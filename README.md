# Microphone field notes

The shareable, dependency-free experiment index is in [`docs/`](docs/index.html); GitHub Pages publishes that directory directly. The first experiment compares seven nearby recordings. The second compares broad-band `speaker` measurement takes with matched `guitar` takes from five devices.

## Rebuild the analysis

`analyze.py` requires Python 3 and NumPy. Update `SOURCE` in the script if the recording folder moves, then run:

```bash
python3 analyze.py
```

It writes the first experiment to `docs/experiments/20260922/data/`. The original files are not modified. The originals are duplicated stereo for six recordings; `sonyPCMD10` has two very similar channels. The analysis folds all to mono so the comparison is consistent. The script's `OUT` constant points at the archived experiment directory.

Run `python3 build_iphone_eq.py` with NumPy available after `analyze.py` to add the derived `iphone2km184.wav` track and its measured curve to `analysis.json`. It uses a 4,097-tap symmetric FIR designed from the smoothed relative iPhone/KM184 spectrum, with correction focused on roughly 70 Hz–4 kHz and limited to +8/−6 dB. The output is LUFS-matched to the other comparison tracks. This is programme-dependent EQ, **not** a microphone emulation or a calibration: residual noise, distortion, DSP, phase, room and directivity differences remain.

Run `python3 build_iphone_iir.py` next to add `iphone2km184-iir.wav` and its measured curve. It fits six broad peaking biquads to the same programme-dependent target, cascades them as a minimum-phase alternative, and LUFS-matches the result. The chosen frequencies, Q values and gains are saved as `eqBands` in `analysis.json`; the FIR and IIR tracks can be compared directly.

The method uses BS.1770-4 K-weighting with absolute and relative gating, FFT cross-correlation for sample offsets, common-overlap trimming, a shared peak-safety gain, and median STFT power ratios smoothed on log-frequency bands. Curves are normalized to the median 500–2,000 Hz level. These are *programme-dependent relative spectral estimates*, not calibrated microphone FR.

### 2026-10-02 speaker / guitar experiment

`analyze_261002.py` reads the five `speaker` and five `guitar` originals from `/Users/sjchoi/dev/rec/261002-iphone,h1,km184,s21,d10`. It requires NumPy and ffmpeg. If the canonical mono PCM cache is absent, set `FFMPEG` when ffmpeg is not on `PATH`, then run:

```bash
FFMPEG=/path/to/ffmpeg python3 analyze_261002.py
```

The script estimates each file's start offset from the RMS envelope, refines it with waveform cross-correlation, fits linear sample-clock drift, and resamples onto the KM184 timeline. Repetitive content that cannot support a trustworthy drift fit falls back to offset-only alignment. It compares corresponding STFT power for the speaker and guitar sets, smooths the relative ratios to 1/6 octave, and writes coverage-weighted curves and agreement metrics to `docs/experiments/20261002/data/analysis.json`.

Only a synchronized, LUFS-matched 90-second guitar excerpt is exported for browser listening. The full source recordings are not copied into the public site. The guitar Solo control uses an 18 ms crossfade and retains the same playhead when switching devices.

`em-usb` exhibits variable latency between different parts of the recording. Its single-offset alignment is approximate; the page shows a sync variation metric and warns listeners. `iphone` and `sonyPCMD10` are polarity-inverted relative to KM184 and are flipped in the comparison exports.

## Publish with GitHub Pages

1. Create a public GitHub repository and push this project (or just `docs/` and the analysis scripts). The two experiments' public assets total about 130 MB.
2. In the repository, open **Settings → Pages** and choose **Deploy from a branch**, your published branch, and **`/docs`** as the folder.
3. Wait for the Pages URL shown there. Every file under `docs/` is then public, including the comparison audio. Do not publish if these recordings contain private content.

The page is static: no server code, external data source, or account is required. The only optional network dependency is a Google Fonts stylesheet; system fonts are used if it is unavailable.

The first listening panel shows the seven microphone recordings plus six iPhone→KM184 FIR, IIR, NAM and TCN-derived comparison tracks with mutually exclusive Solo buttons. It decodes the 13 comparison WAVs once into the browser's audio engine, so switching Solo preserves the common timeline. Set a start and end point with the two sliders or the “current position” buttons, then choose “선택 구간 재생”; the loop checkbox controls whether playback repeats at the end.

Solo switching uses a short 18 ms audio crossfade to reduce clicks. The spectrum view defaults to 70 Hz–4 kHz and can be expanded to the full 40 Hz–18 kHz. Hovering or focusing a microphone legend highlights its curve and shows a coverage-weighted fill underneath. Coverage combines active-vs-quiet spectral contrast, occurrence across active frames, and relative energy; it is a heuristic guide to which frequencies this recording excites, not a calibrated FR confidence interval.

The listening panel shows an overview waveform derived from the aligned KM184 comparison track. A small `docs/experiments/20260922/data/waveform.json` preview loads before the seven WAVs; `python3 build_waveform.py` regenerates it from the comparison export. Click the waveform to seek, or drag across it to set the loop region; the existing start/end sliders remain available for fine adjustment.

The bottom of the first experiment includes a downscaled JPEG of the user-provided recording setup photo at `docs/experiments/20260922/assets/recording-setup.jpg`. It is public when the site is published.

## Interpretation

Without a calibrated sweep/reference, the recordings cannot identify absolute mic frequency response. Room reflections, distance, angle, polar pattern, preamp, DSP, self-noise, and programme content all influence the curves. The most informative relative observations are bass/proximity balance, 2–5 kHz presence, high-frequency roll-off or excess, quiet-passage noise, and whether the relative coloration is consistent over time. Treat 8–16 kHz spikes with particular caution: they can be noise rather than useful acoustic output.
