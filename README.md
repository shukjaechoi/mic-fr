# Seven microphones, one room

Seven WAV recordings from `/Users/sjchoi/dev/mic-fr` are compared by the reproducible `analyze.py` pipeline. The shareable, dependency-free website is in [`docs/`](docs/index.html); GitHub Pages can publish that directory directly.

## Rebuild the analysis

`analyze.py` requires Python 3 and NumPy. Update `SOURCE` in the script if the recording folder moves, then run:

```bash
python3 analyze.py
```

It writes `docs/data/analysis.json` and seven aligned, LUFS-matched, 48 kHz mono WAV files. The original files are not modified. The originals are duplicated stereo for six recordings; `sonyPCMD10` has two very similar channels. The analysis folds all to mono so the comparison is consistent.

The method uses BS.1770-4 K-weighting with absolute and relative gating, FFT cross-correlation for sample offsets, common-overlap trimming, a shared peak-safety gain, and median STFT power ratios smoothed on log-frequency bands. Curves are normalized to the median 500–2,000 Hz level. These are *programme-dependent relative spectral estimates*, not calibrated microphone FR.

`em-usb` exhibits variable latency between different parts of the recording. Its single-offset alignment is approximate; the page shows a sync variation metric and warns listeners. `iphone` and `sonyPCMD10` are polarity-inverted relative to KM184 and are flipped in the comparison exports.

## Publish with GitHub Pages

1. Create a public GitHub repository and push this project (or just `docs/` and `analyze.py`). The `docs/data/` WAV assets total about 47 MB.
2. In the repository, open **Settings → Pages** and choose **Deploy from a branch**, your published branch, and **`/docs`** as the folder.
3. Wait for the Pages URL shown there. Every file under `docs/` is then public, including the comparison audio. Do not publish if these recordings contain private content.

The page is static: no server code, external data source, or account is required. The only optional network dependency is a Google Fonts stylesheet; system fonts are used if it is unavailable.

The listening panel shows seven tracks with mutually exclusive Solo buttons. It decodes the seven comparison WAVs once into the browser's audio engine, so switching Solo preserves the common timeline. Set a start and end point with the two sliders or the “current position” buttons, then choose “선택 구간 재생”; the loop checkbox controls whether playback repeats at the end.

## Interpretation

Without a calibrated sweep/reference, the recordings cannot identify absolute mic frequency response. Room reflections, distance, angle, polar pattern, preamp, DSP, self-noise, and programme content all influence the curves. The most informative relative observations are bass/proximity balance, 2–5 kHz presence, high-frequency roll-off or excess, quiet-passage noise, and whether the relative coloration is consistent over time. Treat 8–16 kHz spikes with particular caution: they can be noise rather than useful acoustic output.
