"""Create supplemental review without changing frozen sources or original export."""
import hashlib
import html
import json
from pathlib import Path

ROOT = Path('/Users/kadir/humanoid-sitting-control')
OUT = ROOT / 'validated_traction_transfer_01/final_results'
v = json.loads((OUT / 'verification.json').read_text())
assert v == json.loads((OUT / 'independent_verified_03/verification.json').read_text())
labels = {'current_reference': 'Older corrected reference', 'current_56211': 'Previous PPO 56211',
          'current_56217': 'Previous PPO 56217', 'bridge_reference': 'Previous bridge reference',
          'bridge_58111': 'Previous bridge PPO 58111', 'bridge_58117': 'Previous bridge PPO 58117',
          'new_reference': 'Repaired traction reference'}
parts = ['''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Reviewed sitting PPO results</title><style>body{font:17px/1.55 system-ui;max-width:1150px;margin:30px auto;padding:0 24px;color:#203345}h1{line-height:1.2}table{border-collapse:collapse;width:100%;margin:20px 0}td,th{text-align:left;border-bottom:1px solid #dce3e7;padding:9px}a{color:#067970}.lead{background:#edf6f4;padding:20px;border-radius:10px}img{width:100%}</style>
<h1>Reliable sitting retained; PPO adds almost no effort saving</h1>
<p class="lead">Both new policies and the repaired reference passed all <b>16/16 unseen-surface trials</b> for 20 seconds, with no recorded slipping, unwanted contacts, falls, impact, joint-limit or torque-limit failures. PPO did not produce a consistent improvement in the primary whole-body squared-torque objective across the two seeds.</p>
<p>The holdout contains eight reserved rigid seats at 38.5–41.5 cm, with slopes from −4° to +4°, each repeated at two placements. All starts are standing, but the reference controls the first 11 seconds; PPO only adjusts the later transfer and rest. No policy was promoted.</p>
<h2>What PPO added to the repaired reference</h2><p>Paired changes on all 16 common full-duration successes; negative means less effort. The primary measure is time-averaged sum of squared physical torque across all 37 joints.</p>
<table><tr><th>New policy</th><th>Whole 20 seconds</th><th>Final 10 seconds</th><th>Final 5 seconds</th></tr>''']
for seed in ['58311', '58317']:
    g = v['comparisons']['holdout_' + seed]['new_reference']['grouped']['all']
    cells = ''.join(f"<td>{g[t]['total_squared_torque_Nm2']['change_percent']:+.3f}%</td>" for t in ['metrics_whole_trial', 'metrics_last10s', 'metrics_last5s'])
    parts.append(f'<tr><td>{seed}</td>{cells}</tr>')
parts.append('''</table><p>Seed 58311 slightly increases squared torque. Seed 58317 reduces it by less than 1%, including across each height and slope group. In the final five seconds, leg torque² changes by +0.329% / −1.117%, while upper-body torque² increases by +0.138% / +0.175%. Absolute torque decreases slightly for both seeds (−0.130% / −0.053% at rest), but this does not establish a meaningful overall improvement. Instantaneous joint RMS averaged over the final five seconds is 2.147 / 2.138 Nm, versus 2.144 Nm for the repaired reference.</p>
<h2>Compared with previous references and policies</h2><p>Each comparison uses its own common-success subset; failed fragments are excluded. The large reductions relative to the older corrected reference and 56211/56217 are already present in the controller repair. Against the previous low-effort bridge policies, reliability improves but resting effort rises.</p>
<table><tr><th>Baseline</th><th>Stable / 16</th><th>Common cases</th><th>58311 whole / final 5s</th><th>58317 whole / final 5s</th></tr>''')
for base, label in labels.items():
    c = [v['comparisons']['holdout_' + seed][base] for seed in ['58311', '58317']]
    cells = ''.join('<td>' + ' / '.join(f"{z['grouped']['all'][t]['total_squared_torque_Nm2']['change_percent']:+.2f}%" for t in ['metrics_whole_trial', 'metrics_last5s']) + '</td>' for z in c)
    parts.append(f"<tr><td>{label}</td><td>{v['evaluations']['holdout_'+base]['successes']}/16</td><td>{len(c[0]['common_success_cases'])}</td>{cells}</tr>")
parts.append('''</table><h2>The remaining grip-margin caveat</h2>
<p>Only <b>11/16 holdout trials are strictly clean</b>. The same five trials in the reference and both PPOs each have one 20 ms touchdown sample above the conservative 0.90 friction-utilization threshold: 39.5 cm at −4° and +4° (both placements), and 40.5 cm at −1° (second placement). The maximum is 0.9485, below modeled capacity 1.0. All flags occur at 10.48–10.54 seconds, before PPO begins at 11 seconds. These are margin flags, not recorded slipping, and have not been erased or relabeled. On the 11 strictly clean common trials, final-five-second torque² changes remain +0.198% / −0.560% versus the repaired reference.</p>
<p>Development: all three new controllers pass 18/18, with 16/18 strictly clean and no slip/contact/physical failures. The two retained 20 ms flags at 42 cm/−5° occur at 10.46 seconds (maximum 0.9140). Previous bridge policies retain two unwanted-contact failures on development; the repaired controller removes them.</p>
<h2>Verification and practical limits</h2>
<p>Independent verification 03 matches 01 and 02 exactly: 76 frozen inputs and 11,342 protected prior/preflight files are unchanged. Both 64-rollout budgets completed: 368,640 total physics transitions = 360,000 active + 8,640 indexed warmup. Active ticks comprise 145,800 learned-control and 214,200 reference-control ticks. Initial settling (900) and development smokes (46,080) are separate. Each seed has 544 actor and 608 critic Adam steps; the critic-only warmup leaves the actor unchanged. Each ends with 180 pending reset-warmup ticks and zero unfinished active ticks. Final actor/critic weights are finite and changed, final checkpoint loads and physical/414-channel sensor startups match exactly, the first 11 seconds match the repaired reference, and evaluations last the full 20 seconds. Fixed fingers, bounded-control implementation and saved command-slew checks pass. Both final checkpoints preceded holdout.</p>
<p>All 1,632 audited holdout geometry observations per new controller are available, with worst plane offset error 2.18 mm and normal error 0.374°. Development retains 58 fresh-depth gaps on the two high-seat trials; stored measured geometry plus joint/IMU tracking remain sensor-only. Inherited base contract files contain older harness labels; traction_contract.json and the frozen traction source define the actual 414-channel, all-nine-surface experiment.</p>
<p>Two seeds and repeated placements on nearby rigid slabs do not establish arbitrary-furniture generalization. Torque is an effort proxy, not electrical energy or proof of a global minimum. The model planner's fallback is physically tested, not a certified geometric clearance. The next research question is how to obtain useful learned effort reductions while retaining this successful approach; no extra training was launched.</p>
<p><a href="findings.html">Original report</a> · <a href="motion_replay.html">Matched motion replay</a> · <a href="independent_verified_03/verification.json">Independent verification</a> · <a href="independent_verified_03/all_tick_review.json">Every-tick traction audit</a></p><img alt="Development and holdout stability and common-success torque comparison" src="comparison.png"></html>''')
(OUT / 'review.html').write_text(''.join(parts))
supplement = ['review.html', 'independent_verified_03/verification.json', 'independent_verified_03/verification_source.py', 'independent_verified_03/all_tick_review.json']
(OUT / 'REVIEW_SHA256.json').write_text(json.dumps({p: hashlib.sha256((OUT / p).read_bytes()).hexdigest() for p in supplement}, indent=2))
print('Supplemental review generated; original 159-file export unchanged.')
