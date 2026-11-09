"""Readable supplement to the immutable, verified experiment export."""
import hashlib
import html
import json
from pathlib import Path
import sys

r = Path(sys.argv[1])
read = lambda p: json.loads(p.read_text())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
v = read(r / 'verification.json')
assert v['passed']
assert (r / 'verification.json').read_bytes() == (r / 'independent_verified_03/verification.json').read_bytes()
original = read(r / 'SHA256.json')
assert all(sha(r / p) == h for p, h in original.items())
seeds = [59111, 59117]
names = {'reference': 'Repaired reference', 'previous_58311': 'Previous PPO 58311', 'previous_58317': 'Previous PPO 58317'}
metric = 'total_squared_torque_Nm2'
def result(seed, baseline, window='metrics_last5s', measure=metric):
    return v['comparisons'][f'holdout_{seed}'][baseline]['grouped']['all'][window][measure]
def pct(value):
    return f'{value:+.2f}%'
def cell(value):
    return f'<td class="{"up" if value > 0 else "down"}">{pct(value)}</td>'

summary_rows = []
comparison_rows = []
surface_rows = []
surfaces = {}
for seed in seeds:
    outcome = v['evaluations'][f'holdout_{seed}']
    assert outcome['successes'] == 16 and outcome['physical_failures'] == 0
    c = v['contacts'][f'holdout_{seed}']
    assert not c['slip_cases'] and not c['unwanted_contact_cases']
    assert c['margin_flags'] == v['contacts']['holdout_reference']['margin_flags']
    assert all(t < 11 for x in c['margin_flags'] for t in x['times_s'])
    summary_rows.append(f'<tr><td>{seed}</td><td>16/16</td><td>0 / 0</td>' + ''.join(cell(result(seed, 'reference', w)['change_percent']) for w in ['metrics_whole_trial', 'metrics_last10s', 'metrics_last5s']) + '</tr>')
    for baseline, name in names.items():
        comparison_rows.append(f'<tr><td>{seed}</td><td>{name}</td>' + ''.join(cell(result(seed, baseline, w)['change_percent']) for w in ['metrics_whole_trial', 'metrics_last10s', 'metrics_last5s']) + '</tr>')
    folder = r / 'evidence/holdout' / str(seed)
    rows = read(folder / 'results.json')['episodes']
    reference = {x['case']: x for x in read(r / 'evidence/holdout/reference/results.json')['episodes']}
    geometry = read(folder / 'geometry.json')
    grouped = {}
    for row in rows:
        g = geometry[row['geometry_index']]
        key = (g['height'], g['pitch_deg'])
        grouped.setdefault(key, []).append((row, reference[row['case']]))
    for key, pairs in grouped.items():
        value = 100 * (sum(a['metrics_last5s'][metric] for a, b in pairs) / sum(b['metrics_last5s'][metric] for a, b in pairs) - 1)
        surfaces.setdefault(key, {})[seed] = value
for (height, slope), values in sorted(surfaces.items()):
    surface_rows.append(f'<tr><td>{height*100:.2f} cm</td><td>{slope:+.1f}°</td><td>2</td>' + ''.join(cell(values[s]) for s in seeds) + '</tr>')

body = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Verified PPO effort comparison</title>
<style>body{font:17px/1.55 system-ui;max-width:1150px;margin:36px auto;padding:0 24px;color:#223743;background:#fafcfc}h1{line-height:1.15}h2{margin-top:32px}table{border-collapse:collapse;width:100%;margin:18px 0;background:white}th,td{padding:10px 12px;border-bottom:1px solid #dce5e5;text-align:left}th{background:#edf3f2}.down{color:#167258}.up{color:#ad352c}.lead{font-size:21px}aside{border-left:4px solid #9a7c36;padding:12px 18px;background:#fff8e9}a{color:#126b78}img{width:100%;height:auto}.scroll{overflow:auto}</style>
<h1>PPO now reduces effort while retaining stable sitting</h1>
<p class="lead">Both new policies passed all 16 unseen-surface trials without slipping, unwanted contacts or physical failures. Their average resting effort fell by <strong>5.20% and 4.57%</strong> versus the repaired reference.</p>
<p>These are learned gains with the same physical reference controller. Both policies also outperform both previous PPO policies on average. All comparisons use matched full 20-second successes. Effort here means the sum of squared physical torque over all 37 joints; negative percentages indicate savings.</p>
<div class="scroll"><table><tr><th>Seed</th><th>Stable</th><th>Slip / unwanted contact cases</th><th>Whole 20s</th><th>Final 10s</th><th>Final 5s resting</th></tr>''' + ''.join(summary_rows) + '''</table></div>
<aside><strong>The existing grip-margin warnings remain.</strong> Eight of 16 trials have a single 20 ms conservative friction-margin flag at 10.46–10.54 seconds, before PPO begins at 11 seconds. The reference and all four policies have the same flags. Maximum friction utilization is 0.9185: above the conservative 0.90 threshold, below modeled capacity 1.0. None slips. Strict clean is 8/16 for every controller; no criterion was relaxed.</aside>
<h2>Compared with each baseline</h2><table><tr><th>New seed</th><th>Baseline</th><th>Whole 20s torque²</th><th>Final 10s</th><th>Final 5s</th></tr>''' + ''.join(comparison_rows) + '''</table>
<p>The smaller whole-motion saving reflects the unchanged first 11 seconds. In the final five seconds, summed absolute torque also decreases by 1.70% and 1.02%. Mean instantaneous all-joint RMS torque falls from 2.106 Nm to 2.052 and 2.050 Nm. Leg torque-squared falls 9.75% and 8.55%; upper-body effort is nearly unchanged. The strict-clean eight-case subset still saves 5.64% and 3.86% in resting torque-squared.</p>
<h2>The improvement is not uniform</h2><p>Seed 59111 improves every held-out placement. Seed 59117 improves 14/16 placements, but uses about 0.35% more resting effort on the 41.75 cm seat tilted −2°. Each row below averages the two repeated placements of one surface.</p>
<table><tr><th>Height</th><th>Slope</th><th>Placements</th><th>59111 vs reference</th><th>59117 vs reference</th></tr>''' + ''.join(surface_rows) + '''</table>
<h2>What was validated</h2><p>Both seeds completed 10 updates and 180 standing-start training episodes, all stable. The total budget is 368100 physics transitions: 360000 active and 8100 indexed warmup, with 900 initial settling ticks separate. Actor Adam steps are 59/49, critic steps 80 each. Three verification passes agree exactly; independent CPU replay reproduces all optimizer updates within documented floating-point tolerances. All 94 frozen input/source hashes and 12898 protected prior/evidence hashes pass. Initializations, final checkpoint loads, starting states and sensor observations, the first 11 seconds, full-duration horizons and held-out timing were checked. The original 128 exported files pass SHA256 verification.</p>
<p>All 1632 audited unseen-surface geometry observations are available. Worst plane offset and normal errors are 2.18 mm and 0.368°. The development bank retains 58 late fresh-depth gaps, with measured-plane memory and kinematics still available.</p>
<h2>What this establishes</h2><p>The revised setup chooses three small persistent hip/knee/ankle adjustments and learns from their complete outcome. The old setup made fleeting, mostly tiny changes with short bootstrapped feedback. This bundled experiment supports a better learning setup; it does not isolate one causal factor.</p><p>This is a small sitting pilot on eight nearby rigid surfaces, each repeated twice, with two training seeds. The reference controls the approach and remains active afterward. It is not unassisted motion discovery, a leaning result, arbitrary-furniture generalization or proof of minimum possible effort. Torque is a proxy, not measured energy. No policy was promoted.</p>
<p><a href="motion_replay.html">Inspect the saved motion</a> · <a href="findings.html">Original comparison</a> · <a href="independent_verified_03/verification.json">Independent audit</a> · <a href="protocol.json">Frozen protocol</a></p><img src="comparison.png" alt="Stability and paired whole-body torque comparisons"></html>'''
(r / 'review.html').write_text(body)
audit = dict(passed=True,original_export_hashes_verified=len(original),independent_verification_matches_exact=True,
             frozen_hashes=v['source_hashes'],prior_hashes=v['prior_hashes'],
             visual_checks=['comparison.png','findings.html','motion_replay.html: unseen 41.75cm/-2deg rest, 39.25cm/-3.5deg transfer and rest'],
             surface_resting_changes=[dict(height_m=h,slope_deg=s,changes_by_seed=c) for (h,s),c in sorted(surfaces.items())])
(r / 'completion_review.json').write_text(json.dumps(audit,indent=2))
(r / 'review_source.py').write_text(Path(__file__).read_text())
manifest = {str(p.relative_to(r)): sha(p) for p in r.rglob('*') if p.is_file() and p.name != 'REVIEW_SHA256.json'}
(r / 'REVIEW_SHA256.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(original_hashes=len(original),supplemental_manifest=len(manifest),review=str(r/'review.html'))))
