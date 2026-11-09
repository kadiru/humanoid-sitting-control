"""Compose final report from verified results, preserving earlier report versions."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

p = argparse.ArgumentParser()
p.add_argument('--verified', required=True)
a = p.parse_args()
r = Path(a.verified)
audit = json.loads((r/'supplementary_audit.json').read_text())
results = json.loads((r/'results.json').read_text())
assert audit['passed'] and json.loads((r/'validation.json').read_text())['passed']
assert all(v['success'] == 0 for v in results.values())
assert all(v['physical_failures'] == (0 if k.startswith('fixed') else results[k]['total'])
           for k, v in audit['evaluations'].items())

labels = [f'eval_{s}_{bank}' for s in [54211,54217] for bank in ['train','holdout']]
fig, axes = plt.subplots(1,2,figsize=(11,4.6))
x = np.arange(4)
for j, (field, title) in enumerate([('camera_samples_including_start','At camera sampling times'),
                                   ('recorded_10Hz_motion_samples','Including tracking between frames')]):
    samples = [audit['sensor'][k][field] for k in labels]
    p95 = [v['offset_abs_p95_mm'] for v in samples]
    maximum = [v['offset_abs_max_mm'] for v in samples]
    axes[j].bar(x,p95,color=['#267b91','#267b91','#d18836','#d18836'],label='95th percentile')
    axes[j].scatter(x,maximum,marker='x',color='#a64236',s=45,label='Maximum')
    axes[j].set_xticks(x,['54211\nTrain','54211\nUnseen','54217\nTrain','54217\nUnseen'])
    axes[j].set(title=title,ylabel='Absolute plane-distance error (mm)',ylim=(0,42))
    axes[j].spines[['top','right']].set_visible(False)
    axes[j].legend(fontsize=9)
fig.suptitle('Tracking before first physical failure: available does not mean exact',fontsize=12)
fig.tight_layout()
fig.savefig(r/'tracking_accuracy.png',dpi=180)

sections = '<h2>Sitting versus leaning</h2><table><tr><th>Seed / bank</th><th>Sitting</th><th>Leaning</th><th>First failure</th></tr>'
for seed, timing in [(54211,'0.76–0.80 s'),(54217,'0.62–0.74 s')]:
    for bank in ['train','holdout']:
        d = results[f'eval_{seed}_{bank}']['by_kind']
        sections += f'<tr><td>{seed} / {bank}</td><td>0/{d["sit"]["total"]}</td><td>0/{d["lean"]["total"]}</td><td>{timing} across both banks</td></tr>'
sections += '</table><p>Every policy trial eventually latched fall, unwanted-contact and joint-limit failures. One seed-54217 training-surface trial also latched an impact failure. No torque-limit failures occurred. Seed 54211 first failed by falling in all 29 cases; seed 54217 first failed by falling in 23 cases and unwanted contact in seven (one overlaps). Later contacts after falling are not evidence of a successful landing.</p>'
sections += '<h2>Geometry tracking during motion</h2><p>All 398 saved pre-failure motion observations were available; no validity flag was lost. However, camera-time statistics alone hide between-frame error: over the recorded 10 Hz motion, the distance-error 95th percentile was 5.46/6.52 mm on training/unseen surfaces for seed 54211, and 23.29/23.93 mm for seed 54217. Maximum error reached 36.08 mm. Plane-normal 95th percentiles stayed below 0.39°. The bridge assumes stationary feet and can drift during slipping or unstable motion. These records cover only the brief interval before failure, not a successful descent; not every 50 Hz control tick was saved.</p><img src="tracking_accuracy.png"><p>The large first-command shifts are already present on the unchanged standing observations, including 16.7°/18.0° after just the first update. This identifies a control-learning problem independently of later tracking drift. It does not rule out tracking error contributing during motion.</p>'

before = r/'review.html'
shutil.copy2(before,r/'diagnosis_review.html')
page = before.read_text()
# Presentation edits only; quantities remain derived from the frozen run.
replacements = {
    'scored<strong>0/13 training surfaces and0/16 unseen surfaces</strong>':'scored <strong>0/13 training surfaces and 0/16 unseen surfaces</strong>',
    'standing in29/29':'standing in 29/29', 'about31° and38°':'about 31° and 38°',
    'affecting17–18':'affecting 17–18', 'roughly30° within0.7seconds':'roughly 30° within 0.7 seconds in the illustrated trial',
    'completed224':'completed 224','full20-second':'full 20-second','initial0.5-second':'initial 0.5-second',
    'and1664-transition':'and 1,664-transition','full20seconds':'full 20 seconds','one20ms':'one 20 ms',
    'endpoints remain useful:13/13':'endpoints remain useful: 13/13','pass20-second':'pass 20-second',
    'separate75°':'separate 75°','pass29/29':'pass 29/29','Vertical90°':'Vertical 90°'
}
for old,new in replacements.items():
    page = page.replace(old,new)
page = page.replace('<h2>Budget and reset accounting</h2>',sections+'<h2>Budget and reset accounting</h2>')
page = page.replace('That is a proposed follow-up; no additional training was launched.',
    'In this pilot, holding the previous command was encoded inside trainable network layers; optimization immediately changed that behavior. A follow-up should protect the reference controller outside those trainable layers, test bounded exploration, and use validated short motion segments before combining full transitions. These are proposed changes, not proven remedies; no additional training was launched.')
page = page.replace('<h2>What remains validated</h2>',
    '<h2>Verification</h2><p>All 36 frozen source/bank hashes and 46 prior-study hashes match. Every run’s saved source copies and bank manifests also match the frozen protocol. Budgets reconcile against episode records, unfinished active steps and pending reset warmup. Both actor and critic weights are finite; final files match the last update and the evaluated actor weights exactly. All four policy/baseline pairs have bitwise-identical initial states, commands, geometry and observations; no tolerance was relaxed. All evaluations contain 1,000 control ticks and 200 recorded motion samples. Fixed commands and uncontrolled joints are constant. No external disturbance schedule was applied in this pilot. Source inspection confirms actor and critic receive the same 333 sensor-derived channels; truth is confined to resets, rewards and evaluation labels.</p><p>The supplementary audit required one reporting-only repair: canonicalizing an equivalent path containing <code>IsaacLab/../</code> before hash lookup. The failed script and recovery manifest are preserved; no research definition or output was changed.</p><h2>What remains validated</h2>')
page = page.replace('<a href="validation.json">Independent verification</a>',
    '<a href="validation.json">Independent verification</a> · <a href="supplementary_audit.json">Source, budget and tracking audit</a> · <a href="report_recovery_01/recovery.json">Reporting repair record</a>')
before.write_text(page)
recovery = r/'report_recovery_01/recovery.json'
record = json.loads(recovery.read_text())
record['repaired_sha256'] = hashlib.sha256((Path(__file__).parent/'audit_supported_transition_delivery.py').read_bytes()).hexdigest()
record['result'] = 'passed; full supplementary audit saved with strict hash and exact startup comparisons'
recovery.write_text(json.dumps(record,indent=2))
print('Report and tracking figure finalized.')
