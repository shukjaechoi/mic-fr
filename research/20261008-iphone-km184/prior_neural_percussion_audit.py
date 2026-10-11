"""Re-tabulate the earlier 10ms MLP experiment around October percussion."""
from pathlib import Path
import json
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / 'research/20261002-guitar-neural/results'
manifest = json.loads((ROOT / 'research/20261002-guitar-linear/dataset/manifest.json').read_text())['segments']
by_id = {x['id']: x for x in manifest}
out = {'scope': 'Post-hoc subset of earlier 2026-10-02 validation segments starting between 1772 and 1800 s; not the newly designed SOS baseline.',
       'method': 'Existing stored segment spectrum (80–4kHz), envelope and onset sums; same .5/.25/.25 score as prior neural experiment. No retraining or parameter selection using this subset.',
       'cautions': ['Seven nearby validation segments from one continuous percussion passage are not seven independent sessions.',
                    'Previous 10ms-block controller has 20ms smoothing and can react late to the initial attack.',
                    'Earlier linear base is iPhone parallel32 / S21 iir12, so scores cannot be subtracted from the current upper-mid SOS score.'],
       'devices': {}}
for device in ('iphone', 's21'):
    results = json.loads((OLD / device / 'validation.json').read_text())['segments']
    device_rows = {}
    for model in sorted({r['model'] for r in results}):
        rows = [r for r in results if r['model'] == model and 1772 < by_id[r['segment']]['start'] / 48000 < 1800]
        if not rows:
            continue
        weight = np.array([r['weight'] for r in rows])
        spectrum = float(np.average([r['spectrum'] for r in rows], weights=weight))
        envelope = sum(r['envelope_sum'] for r in rows) / sum(r['envelope_n'] for r in rows)
        onset = sum(r['onset_sum'] for r in rows) / sum(r['onset_n'] for r in rows)
        device_rows[model] = {'segments': len(rows), 'spectrum_db': round(spectrum, 4),
                              'envelope_db': round(envelope, 4), 'onset_db': round(onset, 4),
                              'score': round(.5 * spectrum + .25 * envelope + .25 * onset, 4)}
    out['devices'][device] = device_rows
p = ROOT / 'docs/experiments/20261008-iphone-km184-public/prior-neural-percussion-audit.json'
p.write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
for device, rows in out['devices'].items():
    print(device, [(key, val['score']) for key, val in rows.items()])
