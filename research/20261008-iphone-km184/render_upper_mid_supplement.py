"""Create synchronized five-second listening/QC page for upper-mid supplements."""
from pathlib import Path
import json
import numpy as np
import soundfile as sf
from scipy import signal, ndimage
from PIL import Image
import matplotlib
matplotlib.use('Agg')
from matplotlib import colormaps

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / 'docs/experiments/20261008-iphone-km184'
PUBLIC = ROOT / 'docs/experiments/20261008-iphone-km184-public'
FS, NF, NT = 48000, 240, 470
config = json.loads((LOCAL / 'upper-mid-supplement.json').read_text())
old = json.loads((ROOT / 'docs/experiments/20261002/data/analysis.json').read_text())
maps = {m['id']: m['guitarSync'] for m in old['microphones']}
maps['iphone'] = json.loads((LOCAL / 'analysis.json').read_text())['sets']['guitar']['mapping']
ref_source = np.load('/private/tmp/mic-audit-20261008/km184-guitar.npy', mmap_mode='r')
sources = {k: np.load(f'/private/tmp/mic-audit-20261008/{k}-guitar.npy', mmap_mode='r') for k in ('iphone', 's21')}
times = {'open_notes': (418, '개방현 단음'), 'complex': (1704, '복합 화음'), 'percussion': (1792, '퍼커션')}
freq_grid = np.geomspace(6000, 80, NF)
magma = colormaps['magma']
div = colormaps['RdBu_r']

def spectrum(y):
    f, t, z = signal.stft(y, FS, nperseg=2048, noverlap=1536, boundary=None)
    db = 20 * np.log10(np.maximum(abs(z), 1e-8))
    target_t = np.linspace(0, 5, NT)
    fi, ti = np.meshgrid(freq_grid / (f[1] - f[0]), (target_t - t[0]) / (t[1] - t[0]), indexing='ij')
    return ndimage.map_coordinates(db, [fi, ti], order=1, mode='nearest').astype('float32')

def save_image(name, rgb):
    Image.fromarray(np.uint8(np.clip(rgb * 255, 0, 255)), 'RGB').save(PUBLIC / name, optimize=True)

manifest = []
for group, (t, label) in times.items():
    start = round(t * FS)
    ref = np.asarray(ref_source[start:start + 5 * FS], float)
    assert len(ref) == 5 * FS
    for device in ('iphone', 's21'):
        m = maps[device]
        ps = round(start * m['slope'] + m.get('intercept_samples', m.get('interceptSamples', 0)))
        left = max(0, ps - round(.25 * FS))
        off = ps - left
        context = np.asarray(sources[device][left:ps + 5 * FS], float)
        raw = context[off:]
        assert len(raw) == 5 * FS
        material = {'km184': ref, 'raw': raw}
        for variant in ('baseline', 'broad_cut', 'shaped'):
            sos = np.asarray(config['models'][device]['sos'][variant], float)
            material[variant] = signal.sosfilt(sos, context)[off:]
        f, rp = signal.welch(ref, FS, nperseg=8192, noverlap=4096)
        _, xp = signal.welch(raw, FS, nperseg=8192, noverlap=4096)
        anchor = np.median([10 * np.log10(max(float(np.mean(rp[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))]) /
                                                      np.mean(xp[(f >= c / 2 ** (1 / 6)) & (f < c * 2 ** (1 / 6))])), 1e-30))
                            for c in (335.7, 597.7)])
        gain = 10 ** (anchor / 20)
        for key in material:
            if key != 'km184':
                material[key] *= gain
        common_scale = min(1., .8 / max(float(np.max(abs(x))) for x in material.values()))
        for key in material:
            material[key] *= common_scale
        specs = {key: spectrum(y) for key, y in material.items()}
        top = max(float(z.max()) for z in specs.values())
        ref_spec = ndimage.gaussian_filter(specs['km184'], sigma=(2, 1.5))
        media = {}
        for key, y in material.items():
            stem = f'uppermid-{group}-{t:04d}-{device}-{key}'
            wav, image = stem + '.wav', stem + '.png'
            sf.write(PUBLIC / wav, y, FS, subtype='PCM_16')
            rgb = magma(np.clip((specs[key] - (top - 72)) / 72, 0, 1))[:, :, :3]
            for sec in range(1, 5):
                rgb[:, round(sec / 5 * (NT - 1)), :] = rgb[:, round(sec / 5 * (NT - 1)), :] * .78 + .22
            for hz in (100, 200, 500, 1000, 2000, 5000):
                row = int(np.argmin(abs(np.log(freq_grid / hz))))
                rgb[row, :, :] = rgb[row, :, :] * .82 + .18
            save_image(image, rgb)
            media[key] = {'audio': wav, 'spectrogram': image}
            if key != 'km184':
                diff = np.clip(ndimage.gaussian_filter(specs[key], sigma=(2, 1.5)) - ref_spec, -12, 12)
                observed = (ref_spec > top - 58) & (specs[key] > top - 58) & (freq_grid[:, None] >= 100) & (freq_grid[:, None] <= 4000)
                ergb = div(np.clip((diff + 12) / 24, 0, 1))[:, :, :3]
                ergb[~observed] = [.88, .90, .91]
                error = stem + '-error.png'
                save_image(error, ergb)
                media[key]['residual'] = error
        manifest.append({'group': group, 'label': label, 'start_s': t, 'device': device,
                         'display_anchor_db': round(float(anchor), 3),
                         'common_display_scale': round(common_scale, 6), 'media': media})
        print('rendered', group, device, flush=True)

payload = {'scope': 'October 2026-10-02; fixed baseline, broad-cut comparator, and two-section shaped supplement',
           'selection': 'Three user-identified phrases at KM184 418, 1704 and 1792 s',
           'display': 'Same raw-derived 335.7/597.7Hz anchor for all device variants, one common safety scale and spectrogram color range within each five-version set. Comparison level is not deployment output gain.',
           'summary': config['summary'], 'response_hz': config['response_hz'],
           'models': {k: {key: config['models'][k][key] for key in ('shaped_section_centers_hz', 'shaped_section_q', 'shaped_section_gains_db', 'broad_cut', 'sos', 'sha256', 'response_db')} for k in ('iphone', 's21')},
           'implementation_note': '48kHz causal SOS. For offline trial, start with -3dB output attenuation after the full filter; selected five-second preview scale is separate. Full-recording intersample peaks and other input levels have not been validated.',
           'clips': manifest}
(PUBLIC / 'upper-mid-supplement-browser.json').write_text(json.dumps(payload, ensure_ascii=False, indent=2) + '\n')
