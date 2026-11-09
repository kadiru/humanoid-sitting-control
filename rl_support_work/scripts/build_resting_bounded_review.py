"""Build the local, human-readable closeout from independently audited data."""
import html
import json
from pathlib import Path

ROOT=Path('/Users/kadir/humanoid-sitting-control')
OUT=ROOT/'validated_resting_bounded_02/final_results'
audit=json.loads((OUT/'verified_review/validation.json').read_text())
def pct(x):return f'{x:+.2f}%'
def table(headers,rows):
    return '<div class="table"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join('<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table></div>'
all_rows=[r for r in audit['comparisons'] if r['bank']=='all']
intro=[['Fixed posture','56/56','Reference','Reference','Reference','Reference']]
for r in all_rows:
    m=r['metrics'];intro.append([f"PPO {r['seed']}",'56/56',pct(m['normalized_torque_squared']['percent_change']),pct(m['all_joint_rms_Nm']['percent_change']),pct(m['leg_rms_Nm']['percent_change']),pct(m['torso_arm_hand_rms_Nm']['percent_change'])])
bank_names={'starts':'Original heights','heights':'Intermediate heights','impulse':'Velocity impulses','novel':'New holdout heights'}
bank_rows=[]
for bank,label in bank_names.items():
    rs=[next(r for r in audit['comparisons'] if r['bank']==bank and r['seed']==s) for s in (53111,53117)]
    bank_rows.append([label,f"{rs[0]['total']}/{rs[0]['total']}",pct(rs[0]['metrics']['normalized_torque_squared']['percent_change']),pct(rs[1]['metrics']['normalized_torque_squared']['percent_change'])])
height_rows=[]
for h in sorted({r['height_m'] for r in audit['by_height']}):
    rs=[next(r for r in audit['by_height'] if r['height_m']==h and r['seed']==s and r['bank']!='impulse') for s in (53111,53117)]
    height_rows.append([f'{h*100:g} cm','4/4 per policy',pct(rs[0]['metrics']['normalized_torque_squared']['percent_change']),pct(rs[1]['metrics']['normalized_torque_squared']['percent_change'])])
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Bounded resting PPO: verified comparison</title>
<style>body{font:17px/1.65 system-ui;color:#203d46;max-width:1160px;margin:42px auto;padding:0 24px}h1{font-size:36px;line-height:1.2;max-width:950px}h2{font-size:25px;margin-top:34px}p{max-width:1050px}.result{background:#eaf4f0;border-left:4px solid #398570;padding:20px 25px}.finding{background:#fff5e9;border-left:4px solid #b86b35;padding:18px 25px}.note{font-size:15px;color:#536870}img{width:100%;height:auto}table{border-collapse:collapse;width:100%;font-size:15px}th,td{text-align:left;padding:11px 10px;border-bottom:1px solid #d4dfe3}th{background:#edf3f5}.table{overflow:auto}a{color:#1d6e90}code{font-size:14px;overflow-wrap:anywhere}li{margin:8px 0}</style><body>
<p class="note">Verified October 12, 2026 · Finished 11:39 a.m. PDT · Seated-resting PPO pilot 02</p>
<h1>Stable sitting is retained. Lower-effort sitting has not been learned.</h1>
<p class="result"><strong>Both trained PPO policies passed all 56 twenty-second conditions.</strong> Fixed posture also passed 56/56. There were no falls, unwanted contacts, torque-limit, impact-limit or joint-margin failures in these evaluations. Both hold-enabled variants produced exactly the same recorded trajectories and results as their corresponding plain PPO policy.</p>
<p class="finding"><strong>The primary effort measure worsened by 0.28% and 2.59%.</strong> No trial achieved the declared 10% reduction target. The optional hold rule never activated. Neither policy is promoted over fixed posture.</p>
'''
page+=table(['Controller','Success','Normalized squared torque','Whole-body RMS torque','Leg RMS torque','Torso/arm/hand RMS'],intro)
page+='''<p class="note">Changes are relative to the matched fixed-posture trials; negative means lower effort. All 56 trials are common successes. The effort summaries average qualifying ticks. The Nm metric is a time-average of instantaneous joint RMS; normalized squared torque weights each joint by its existing effort limit. Neither is measured electrical energy. Different weighting explains why tiny Nm reductions can coexist with a higher normalized objective.</p>
<img src="comparison.png" alt="All five evaluation variants pass56 trials, but neither PPO seed reaches the10 percent normalized-effort reduction target">
<h2>Across seat heights and disturbances</h2>
<p>Both policies pass all conditions at each tested height, including the four new heights. However, the normalized effort measure increases at every height. For seed 53117, the relative penalty reaches 24.10% at 26.25 cm. Low seats have smaller baseline values, so these relative increases are much larger than the pooled mean suggests.</p>
'''
page+=table(['Bank','Success for each controller','Seed 53111 effort change','Seed 53117 effort change'],bank_rows)
page+='<img src="verified_review/effort_by_height.png" alt="Effort by seat height and evaluation bank, showing increased normalized effort in both seeds">'
page+=table(['Undisturbed seat height','Success','Seed 53111 effort change','Seed 53117 effort change'],height_rows)
page+='''<h2>Why “relax and hold” did not help yet</h2>
<p>The rule requires filtered measured effort to stay below 90% of its initial settled value for two seconds, with low motion and valid nearby depth geometry. There were zero entries or releases in the full-rate gate counters. In the saved 10 Hz samples, effort never reached that threshold: the lowest recorded ratio was 91.4% for seed 53111 during a disturbance and 99.1% for seed 53117.</p>
<p>Turning the rule on therefore changed nothing. This run does not establish whether the rule helps or hurts after reaching a low-effort pose. It only shows that these policies never supplied the required situation. The gate uses an initial measured reference; the study's 10% saving goal instead compares successful-trial averages with fixed posture.</p>
<h2>What the robot actually changed</h2>
<p>The largest requested joint offsets were only 0.314° and 0.415° from the starting commands. Both policies stayed close to the fixed pose. The first seed's largest adjustment was a shoulder-roll target; the second's was the right knee. The robot did not substantially shift its weight onto the seat: average foot support increased slightly, from 41.89% of body weight to 42.01% and 42.66%.</p>
<p>This is consistent with conservative behavior, but it does not identify a unique cause for weak effort learning. This pilot cannot distinguish limited exploration, short training, reward sensitivity, or the absence of a useful lower-effort posture nearby. The previous drifting policies used different action and observation contracts, seeds and budgets, so the old-to-new comparison is not an isolated ablation.</p>
<h2>Verified execution</h2>
<ul><li>Two seeds × 64 updates × 64 steps × 16 environments = <strong>131,072 transitions</strong>: 127,340 active and 3,732 reset-warmup transitions. Completed and unfinished episode accounting reconciles.</li>
<li>Finite changed actor and critic weights, zero initial mean actions, and exact intended final checkpoints loaded for both evaluation variants.</li>
<li>All 16 learned-evaluation startup arrays match fixed posture bit for bit, including the sensor observations. Geometry and disturbance schedules match.</li>
<li>All trials run 1,000 ticks / 20 seconds; fixed targets are constant, learned targets stay bounded, and non-controlled targets remain fixed. Sensor acquisition does not advance the physics clock.</li>
<li>All 46 frozen source/data/protocol hashes remain unchanged. Four controller tests, reset isolation, ten settled reset cycles and prior zero-action retention checks passed.</li>
<li>All 16 new held-out captures are valid. Both trained policies have zero physical failure flags across their 56 conditions.</li></ul>
<h2>Scope and next decision</h2>
<p>There are 44 unique reset states, plus 12 repeats with velocity impulses. The same states are reused across the two training seeds and hold variants. These results cover rigid level seats at 11 heights from 25 to 40 cm; they do not establish performance on cushions, tilted supports, arbitrary furniture, or hardware.</p>
<p>The next useful experiment would first measure whether slow, deliberate changes around these stable poses can actually lower effort while maintaining legal support. That would distinguish a posture/search problem from an RL learning problem and provide a reachable effort target. This is a proposal only: no additional training or policy promotion was started.</p>
<p><a href="verified_review/validation.json">Independent verification and per-height metrics</a> · <a href="results.json">Original complete study results</a> · <a href="protocol.json">Frozen protocol</a> · <a href="../diagnosis/review.html">Contact diagnosis and retention validation</a></p>
<p class="note">Original Ubuntu study: <code>/data/humanoid-sitting-perception/resting_bounded_ppo_02</code>. All earlier studies, model checkpoints, failed probe evidence and original reports are preserved.</p></body></html>'''
(OUT/'review.html').write_text(page)
print(OUT/'review.html')
