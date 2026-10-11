"""Controlled local 1/3, 1/6 and 1/12-octave EQ comparison.

Freeze the last heard filters, add one source-supported S21 high shelf, and
compare fractional-octave middle-band estimates AND actual causal peak widths.
Same-session leave-technique-out checks select resolution; the inspected final
piece is kept out of new curve estimation. Source-derived masks are shared.
"""
from pathlib import Path
import json, hashlib
import numpy as np
from scipy import signal, optimize
import soundfile as sf

ROOT = Path(__file__).resolve().parents[2]
LOCAL = ROOT / 'docs/experiments/20261008-iphone-km184'
OUT = ROOT / 'docs/experiments/20261008-iphone-km184-public'
FS, N = 48000, 8192
prior = json.loads((LOCAL / 'speaker-guitar-confidence-eq.json').read_text())
bright = json.loads((LOCAL / 'brightness-extension.json').read_text())
four = json.loads((LOCAL / 'october-four-device-eq.json').read_text())
bg = json.loads((LOCAL / 'guitar-background-v2.json').read_text())
old = json.loads((ROOT / 'docs/experiments/20261002/data/analysis.json').read_text())
an = json.loads((LOCAL / 'analysis.json').read_text())
maps = {m['id']: m['guitarSync'] for m in old['microphones']}
smaps = {m['id']: m['speakerSync'] for m in old['microphones']}
maps['iphone'] = an['sets']['guitar']['mapping']; smaps['iphone'] = an['sets']['speaker']['mapping']
audio = {d: np.load(f'/private/tmp/mic-audit-20261008/{d}-guitar.npy', mmap_mode='r') for d in ('km184', 'iphone', 's21')}
current = {'iphone': np.array(prior['models']['iphone']['sos']), 's21': np.array(bright['models']['s21']['sos'])}
f = np.fft.rfftfreq(N, 1 / FS)
def mapped(m, t): return round(t * FS * m['slope'] + m.get('interceptSamples', m.get('intercept_samples', 0)))
def psd(x): return signal.welch(np.asarray(x, float), FS, nperseg=N, noverlap=N // 2)[1]
def context(d, t):
    st = mapped(maps[d], t); left = max(0, st - FS // 4)
    return np.asarray(audio[d][left:st + 5 * FS], float), st - left
def mask(c, octaves): return (f >= c / 2 ** (octaves / 2)) & (f < c * 2 ** (octaves / 2))
def meanbands(p, mm): return np.array([np.mean(p[m]) for m in mm])
def response(sos, hz):
    if len(sos) == 0: return np.zeros_like(hz, dtype=float)
    return 20 * np.log10(np.maximum(abs(signal.sosfreqz(sos, worN=hz, fs=FS)[1]), 1e-12))
def peak(fc, q, gain):
    a = 10 ** (gain / 40); w = 2 * np.pi * fc / FS; alpha = np.sin(w) / (2 * q); c = np.cos(w)
    b = np.array([1 + alpha * a, -2 * c, 1 - alpha * a]); z = np.array([1 + alpha / a, -2 * c, 1 - alpha / a])
    return np.r_[b / z[0], z / z[0]]
def shelf(fc, gain):
    a = 10 ** (gain / 40); w = 2 * np.pi * fc / FS; c = np.cos(w); z = 2 * np.sqrt(a) * np.sin(w) / np.sqrt(2)
    b = a * np.array([(a + 1) + (a - 1) * c + z, -2 * ((a - 1) + (a + 1) * c), (a + 1) + (a - 1) * c - z])
    den = np.array([(a + 1) - (a - 1) * c + z, 2 * ((a - 1) - (a + 1) * c), (a + 1) - (a - 1) * c - z])
    return np.r_[b / den[0], den / den[0]]
floor = {d: np.array(bg['median_psd'][d]) for d in ('km184', 'iphone')}
floor['s21'] = np.median([psd(audio['s21'][mapped(maps['s21'], q['start_s']):mapped(maps['s21'], q['start_s'] + 2)]) for q in bg['selected']], axis=0)
anchors = [mask(c, 1 / 3) for c in (335.7, 597.7)]
highmask = (f >= 8000) & (f < 12000)
common = 1000 * 2 ** (np.arange(-5, 14) / 12)
common = common[((common >= 780) & (common <= 1030)) | ((common >= 1450) & (common <= 2050))]
eval_masks = [mask(c, 1 / 12) for c in common]
regions = {'800_1000': np.flatnonzero(common <= 1030), '1500_2000': np.flatnonzero(common >= 1450)}
outside_c = np.array([150, 300, 500, 600, 2700, 3500, 5000.])
outside_masks = [mask(c, 1 / 6) for c in outside_c]

def confidence(r, mm):
    kp, xp = meanbands(r['ref_psd'], mm), meanbands(r['raw_psd'], mm)
    kf, xf = meanbands(floor['km184'], mm), meanbands(floor[r['device']], mm)
    snr = np.minimum(10 * np.log10(np.maximum(kp / kf, 1e-30)), 10 * np.log10(np.maximum(xp / xf, 1e-30)))
    snr = np.minimum(snr, r['anchor_min_snr_db'])
    # Relevance uses integrated energy relative to the full useful 100–12kHz
    # octave map, rather than judging narrow harmonic gaps as strong evidence.
    df = f[1] - f[0]
    strongest_k = max(np.sum(r['ref_psd'][(f >= lo) & (f < hi)]) * df for lo, hi in [(100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6400),(6400,12000)])
    strongest_x = max(np.sum(r['raw_psd'][(f >= lo) & (f < hi)]) * df for lo, hi in [(100,200),(200,400),(400,800),(800,1600),(1600,3200),(3200,6400),(6400,12000)])
    # Scale a fractional band to its corresponding octave footprint so finer
    # estimators do not lose confidence just by having narrower integration.
    centers = np.array([np.mean(f[m]) for m in mm])
    importance = np.minimum(10 * np.log10(np.maximum(kp * centers * .7071 / strongest_k, 1e-30)),
                            10 * np.log10(np.maximum(xp * centers * .7071 / strongest_x, 1e-30)))
    weights = np.clip((snr - 10) / 15, 0, 1) * (.2 + .8 * np.clip((importance + 36) / 24, 0, 1))
    return weights, snr

records = {d: [] for d in current}
for d in current:
    starts = [(g, r['start_s'], False) for g, rr in prior['windows'][d].items() for r in rr]
    starts += [('piece', r['start_s'], True) for r in four['piece_same_session']['windows_data'] if d in r['devices']]
    for g, t, last in starts:
        ref = np.asarray(audio['km184'][round(t * FS):round((t + 5) * FS)], float); x, off = context(d, t)
        if len(ref) != 5 * FS or len(x[off:]) != 5 * FS: continue
        rp, xp = psd(ref), psd(x[off:]); ap, ax = meanbands(rp, anchors), meanbands(xp, anchors)
        anchor = float(np.median(10 * np.log10(ap / ax)))
        asnr = min(*list(10 * np.log10(ap / meanbands(floor['km184'], anchors))), *list(10 * np.log10(ax / meanbands(floor[d], anchors))))
        r = {'device': d, 'group': g, 'start_s': t, 'last_piece': last, 'ref_psd': rp,
             'raw_psd': xp, 'current_psd': psd(signal.sosfilt(current[d], x)[off:]),
             'anchor_db': anchor, 'anchor_min_snr_db': asnr}
        w, snr = confidence(r, eval_masks); r['eval_weight'] = w; r['eval_snr'] = snr
        records[d].append(r)

# Strong residual above 8k is a broad, source-supported effect, not the target
# of the 1/3-vs-1/6-vs-1/12 middle-band comparison. Freeze this same extra shelf
# for all three S21 resolution candidates so those comparisons isolate width.
speaker_audio = {d: np.load(f'/private/tmp/mic-audit-20261008/{d}-speaker.npy', mmap_mode='r') for d in ('km184', 'iphone')}
speaker_audio['s21'], sr = sf.read('/Users/sjchoi/dev/rec/261002-iphone,h1,km184,s21,d10/s21 speaker.wav')
assert sr == FS
speaker_high = {}; speaker_mid = {d:{} for d in current}
for name, a, b in [('pink',252,262),('white',219,232),('log',525.2,543.9)]:
    pp = {d: psd(v[mapped(smaps[d], a):mapped(smaps[d], b)]) for d, v in speaker_audio.items()}
    am = (f >= 300) & (f < 700)
    speaker_high[name] = float(10 * np.log10(np.mean(pp['km184'][highmask]) / np.mean(pp['s21'][highmask])) - 10 * np.log10(np.sum(pp['km184'][am]) / np.sum(pp['s21'][am])))
    for d in current:
        off = float(np.median(10*np.log10(meanbands(pp['km184'],anchors)/meanbands(pp[d],anchors))))
        predicted = pp[d] * 10**(response(current[d],f)/10)
        speaker_mid[d][name] = (10*np.log10(meanbands(pp['km184'],eval_masks)/meanbands(predicted,eval_masks))-off).round(3).tolist()
high_records = []
for r in records['s21']:
    if r['last_piece']: continue
    w, snr = confidence(r, [highmask])
    if w[0] < .25: continue
    val = float(10 * np.log10(np.mean(r['ref_psd'][highmask]) / np.mean(r['current_psd'][highmask])) - r['anchor_db'])
    high_records.append({'group': r['group'], 'start_s': r['start_s'], 'desired_db': val, 'weight': float(w[0])})
high_bygroup = [np.median([r['desired_db'] for r in high_records if r['group'] == g]) for g in sorted({r['group'] for r in high_records})]
assert len(high_bygroup) >= 3 and np.median(high_bygroup) < 0 and all(x < 0 for x in speaker_high.values())
unit = float(np.mean(response(np.array([shelf(8000, -1)]), np.geomspace(8000, 12000, 80))))
highgain = float(np.clip(np.median(high_bygroup) / abs(unit), -6, 0))
high_only = {'iphone': current['iphone'], 's21': np.vstack([current['s21'], shelf(8000, highgain)])}
for d in records:
    ratio = 10 ** ((response(high_only[d], f) - response(current[d], f)) / 10)
    for r in records[d]: r['base_psd'] = r['current_psd'] * ratio

def estimate(d, res, excluded=None):
    nodes = 1000 * 2 ** (np.arange(-12, 24) / res)
    nodes = nodes[((nodes >= 700) & (nodes <= 1200)) | ((nodes >= 1350) & (nodes <= 2250))]
    mm = [mask(c, 1 / res) for c in nodes]
    groups = sorted({r['group'] for r in records[d] if not r['last_piece'] and r['group'] != excluded})
    group_values, group_weights = [], []
    for g in groups:
        vals, weights = [], []
        for r in records[d]:
            if r['group'] != g: continue
            w, _ = confidence(r, mm)
            vals.append(10 * np.log10(meanbands(r['ref_psd'], mm) / meanbands(r['base_psd'], mm)) - r['anchor_db'])
            weights.append(w)
        vals, weights = np.array(vals), np.array(weights)
        group_values.append([np.median(vals[:, j][weights[:, j] >= .15]) if np.any(weights[:, j] >= .15) else np.nan for j in range(len(nodes))])
        group_weights.append(np.median(weights, axis=0))
    values, weights = np.array(group_values), np.array(group_weights)
    target, reliability, support, sign_fraction, dispersion = [], [], [], [], []
    for j in range(len(nodes)):
        vv = values[:, j]; vv = vv[np.isfinite(vv)]; n = len(vv)
        med = float(np.median(vv)) if n else 0.; spread = float(np.median(abs(vv - med))) if n else 0.
        fraction = float(np.mean(np.sign(vv) == np.sign(med))) if n else 0.
        rel = min(n / 5, 1) * max(0, 2 * fraction - 1) * float(np.median(weights[:, j])) / (1 + spread / 6) if n >= 3 else 0.
        target.append(float(np.clip(med, -4, 4))); reliability.append(rel); support.append(n); sign_fraction.append(fraction); dispersion.append(spread)
    target, rel = np.array(target), np.array(reliability)
    q = float(np.sqrt(2 ** (1 / res)) / (2 ** (1 / res) - 1))
    unit = np.stack([response(np.array([peak(c, q, 1)]), nodes) for c in nodes], axis=1)
    dense = np.geomspace(650, 2600, 49); h = np.log2(dense[1] / dense[0])
    shape = np.stack([response(np.array([peak(c, q, 1)]), dense) for c in nodes], axis=1)
    curvature = np.diff(shape, n=2, axis=0) / h ** 1.5
    outer = np.array([100, 300, 500, 600, 1250, 2700, 3500, 6000, 10000.])
    outside = np.stack([response(np.array([peak(c, q, 1)]), outer) for c in nodes], axis=1)
    # All resolutions share physical-frequency curvature penalty, a zero
    # response outside the requested bands, and ±4dB section/response bounds.
    A = np.vstack([unit * np.sqrt(rel[:, None]), .002 * curvature, .65 * outside, .12 * np.eye(len(nodes))])
    b = np.r_[target * np.sqrt(rel), np.zeros(len(A) - len(nodes))]
    gains = optimize.lsq_linear(A, b, bounds=(-4,4)).x
    extra = np.array([peak(c, q, g) for c, g in zip(nodes, gains)])
    for _ in range(3):
        mx = max(abs(response(extra, np.geomspace(500, 3500, 600))))
        if mx <= 4.001: break
        gains *= 4 / mx; extra = np.array([peak(c, q, g) for c, g in zip(nodes, gains)])
    return extra, {'nodes_hz': nodes.tolist(), 'band_octaves': 1 / res, 'peak_q': q,
                   'section_gains_db': gains.tolist(), 'target_db': target.tolist(),
                   'reliability': rel.tolist(), 'support_groups': support,
                   'group_sign_fraction': sign_fraction, 'group_mad_db': dispersion}

models, cv, final_piece = {}, {}, {}
def score(r, added):
    pred = r['base_psd'] * 10 ** (response(added, f) / 10)
    err = 10 * np.log10(meanbands(r['ref_psd'], eval_masks) / meanbands(pred, eval_masks)) - r['anchor_db']
    results = {}
    for name, ii in {'mid_all': np.arange(len(common)), **regions}.items():
        valid = ii[r['eval_weight'][ii] >= .15]
        results[name] = float(np.average(abs(err[valid]), weights=r['eval_weight'][valid])) if len(valid) else None
    return results
def baseline_score(r):
    original = r['base_psd']; r['base_psd'] = r['current_psd']
    result = score(r, np.empty((0,6))); r['base_psd'] = original
    return result
for d in current:
    models[d] = {}; cv[d] = {}; final_piece[d] = {}
    groups = sorted({r['group'] for r in records[d] if not r['last_piece']})
    for res in (3,6,12):
        key = f'oct_{res}'; extra, detail = estimate(d, res)
        sos = np.vstack([high_only[d], extra]); models[d][key] = {'sos': sos.tolist(), 'extra_sos': extra.tolist(), 'design': detail,
            'sha256': hashlib.sha256(np.asarray(sos,dtype='<f8').tobytes()).hexdigest(),
            'stable': all(max(abs(np.roots(section[3:]))) < 1 for section in sos)}
        rr = []
        for g in groups:
            excluded_extra, _ = estimate(d, res, g)
            for r in records[d]:
                if r['group'] == g:
                    rr.append({'group': g, 'start_s': r['start_s'], 'before': baseline_score(r), 'after': score(r, excluded_extra),
                               'weight': r['eval_weight'].round(3).tolist()})
        bygroup = {}
        for g in groups:
            subset = [r for r in rr if r['group'] == g]
            bygroup[g] = {}
            for region in ('mid_all','800_1000','1500_2000'):
                keep = [r for r in subset if r['after'][region] is not None]
                bygroup[g][region] = {'windows':len(keep), 'before': float(np.mean([r['before'][region] for r in keep])) if keep else None,
                                      'after': float(np.mean([r['after'][region] for r in keep])) if keep else None}
        valid = [r for r in rr if r['after']['mid_all'] is not None]
        cv[d][key] = {'equal_group_mid_mae_db': float(np.mean([g['mid_all']['after'] for g in bygroup.values() if g['mid_all']['after'] is not None])),
                      'baseline_equal_group_mid_mae_db': float(np.mean([g['mid_all']['before'] for g in bygroup.values() if g['mid_all']['before'] is not None])),
                      'worse_windows_gt_0_5db': sum(r['after']['mid_all'] > r['before']['mid_all'] + .5 for r in valid),
                      'valid_windows':len(valid), 'groups':bygroup, 'windows':rr}
        pp = [score(r,extra) for r in records[d] if r['last_piece']]
        final_piece[d][key] = {region: float(np.mean([r[region] for r in pp if r[region] is not None])) if any(r[region] is not None for r in pp) else None for region in ('mid_all','800_1000','1500_2000')}
    final_piece[d]['current'] = {region: float(np.mean([baseline_score(r)[region] for r in records[d] if r['last_piece'] and baseline_score(r)[region] is not None])) for region in ('mid_all','800_1000','1500_2000')}
    # Predeclared selection: retain coarse unless the finer resolution gains
    # >=.2dB in leave-technique-out MAE and worsens no >5 percentage points more.
    chosen = 'oct_3'
    for key in ('oct_6','oct_12'):
        best, new = cv[d][chosen], cv[d][key]
        if best['equal_group_mid_mae_db'] - new['equal_group_mid_mae_db'] >= .2 and new['worse_windows_gt_0_5db'] / new['valid_windows'] <= best['worse_windows_gt_0_5db'] / best['valid_windows'] + .05:
            chosen = key
    if cv[d][chosen]['equal_group_mid_mae_db'] >= cv[d][chosen]['baseline_equal_group_mid_mae_db'] - .15:
        chosen = 'high_only' if d == 's21' else 'current'
    models[d]['selected'] = chosen
    models[d]['application_scope'] = {'tonal_option':chosen,'percussion_option':'high_only' if d=='s21' else 'current',
        'reason':'Fine middle correction improves tonal group averages but worsens percussion middle balance; preserve the earlier middle filter for percussion. This is manual profile selection, not an automatic technique classifier.'}
    selected_sos = np.array(models[d][chosen]['sos']) if chosen.startswith('oct_') else high_only[d]
    zi = np.zeros((len(selected_sos),2)); maximum = 0.
    for start in range(0,len(audio[d]),10*FS):
        y,zi = signal.sosfilt(selected_sos,np.asarray(audio[d][start:start+10*FS],float),zi=zi)
        maximum = max(maximum,float(np.max(abs(y))))
    models[d]['selected_full_sample_peak_dbfs_before_output_gain'] = float(20*np.log10(maximum))
    models[d]['output_gain_db'] = -3.0 if d == 'iphone' else -6.5

# Validate the spectrum-domain score approximation against actual context-aware
# causal filtering of the exact preview phrases, and preserve their signed bands.
clips = []
for d in current:
    for g,t,label in [('arpeggio',100,'아르페지오'),('open_notes',418,'개방현 단음'),('complex',1704,'복합 화음'),('percussion',1792,'퍼커션'),('piece',1930,'마지막 곡')]:
        x,off=context(d,t);ref=np.asarray(audio['km184'][t*FS:(t+5)*FS],float)
        rp,xp=psd(ref),psd(x[off:]);ap,ax=meanbands(rp,anchors),meanbands(xp,anchors)
        r={'device':d,'ref_psd':rp,'raw_psd':xp,'anchor_db':float(np.median(10*np.log10(ap/ax))),
           'anchor_min_snr_db':min(*list(10*np.log10(ap/meanbands(floor['km184'],anchors))),*list(10*np.log10(ax/meanbands(floor[d],anchors))))}
        weight,snr=confidence(r,eval_masks);r['eval_weight']=weight
        hw,hsnr=confidence(r,[highmask])
        r['current_psd']=psd(signal.sosfilt(current[d],x)[off:]);r['base_psd']=psd(signal.sosfilt(high_only[d],x)[off:])
        results={};maxgap=0.
        variants={'current':current[d],**({'high_only':high_only[d]} if d=='s21' else {}),**{key:np.array(models[d][key]['sos']) for key in ('oct_3','oct_6','oct_12')}}
        for key,sos in variants.items():
            actual=psd(signal.sosfilt(sos,x)[off:]);err=10*np.log10(meanbands(rp,eval_masks)/meanbands(actual,eval_masks))-r['anchor_db']
            higherr=float(10*np.log10(np.mean(rp[highmask])/np.mean(actual[highmask]))-r['anchor_db'])
            scores={region:float(np.average(abs(err[ii][weight[ii]>=.15]),weights=weight[ii][weight[ii]>=.15])) if np.any(weight[ii]>=.15) else None for region,ii in {'mid_all':np.arange(len(common)),**regions}.items()}
            if key.startswith('oct_'):
                predicted=score(r,np.array(models[d][key]['extra_sos']))
                if scores['mid_all'] is not None:maxgap=max(maxgap,abs(scores['mid_all']-predicted['mid_all']))
            results[key]={'scores':scores,'signed_km_minus_device_db':err.round(3).tolist(),'8000_12000_signed_db':round(higherr,3)}
        clips.append({'group':g,'label':label,'start_s':t,'device':d,'anchor_db':r['anchor_db'], 'eval_confidence':weight.round(3).tolist(),'eval_joint_snr_db':snr.round(2).tolist(),
                      'high_confidence':float(hw[0]),'high_joint_snr_db':float(hsnr[0]),
                      'results':results,'max_actual_prediction_score_gap_db':maxgap})
assert max(c['max_actual_prediction_score_gap_db'] for c in clips) < .1, 'Predicted and actual preview scores do not agree within 0.1dB'

hz=np.geomspace(100,16000,480)
report={'scope':'October same-session controlled local-resolution experiment; September unused',
        'user_target_regions_hz':[[800,1000],[1500,2000],[8000,12000]],'common_evaluation_centers_hz':common.tolist(),
        'rules':{'resolutions':[3,6,12], 'fit_regions_hz':[[700,1200],[1350,2250]],'section_gain_limit_db':4,'correction_response_limit_db':4,
                 'cross_validation':'Supplementary EQ excludes the evaluation technique; frozen prior filters already saw this session, so this is a development stability check.',
                 'confidence':'Joint raw SNR over background, adequate fixed anchors and per-octave-scaled energy relevance. Group coverage, sign consistency and MAD lower fitting confidence.',
                 'regularization':'Same physical-frequency curvature, outside-zero response and coefficient penalties for all grids. Filter peak bandwidth matches estimation octave width; all score variants use common 1/12 octave masks.',
                 'implementation_audit':'Initial physical-curvature coefficient 0.06 numerically suppressed almost all fine-filter gains (<0.1dB for most 1/12 sections); it was changed to 0.002 before the meaningful resolution comparison. This remains an exploratory development study, not a preregistered independent experiment.',
                 'selection':'Finer must gain >=0.2dB LOTO group MAE without >5 percentage-point extra worsening. New middle EQ must gain >=0.15dB versus frozen current. Last piece omitted from fit and candidate choice.'},
        's21_high_shelf':{'center_hz':8000,'slope':1,'extra_gain_db':highgain,'speaker_km_minus_device_8_12k_db':speaker_high,
                         'reliable_development_windows':len(high_records),'group_median_needed_gain_db':float(np.median(high_bygroup)), 'high_records':high_records},
        'speaker_current_mid_km_minus_device_db':speaker_mid,
        'models':models,'cv':cv,'last_piece':final_piece,'clips':clips,'response_hz':hz.tolist(),
        'response_db':{d:{'current':response(current[d],hz).round(4).tolist(),**({'high_only':response(high_only[d],hz).round(4).tolist()} if d=='s21' else {}),**{key:response(np.array(models[d][key]['sos']),hz).round(4).tolist() for key in ('oct_3','oct_6','oct_12')}} for d in current}}
(LOCAL/'local-resolution-eq.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps({d:{'selected':models[d]['selected'],'cross_validation':{k:{n:v[n] for n in ('equal_group_mid_mae_db','baseline_equal_group_mid_mae_db','worse_windows_gt_0_5db','valid_windows')} for k,v in cv[d].items()},'last_piece':final_piece[d]} for d in current},ensure_ascii=False,indent=2),flush=True)
