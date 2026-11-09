"""Generate the research interpretation and a preserved, portable evidence copy."""
from pathlib import Path
import hashlib
import html
import json
import shutil

r=Path('/data/humanoid-sitting-perception/two_surface_curriculum_01')
study=r/'ppo_01';report=r/'final_verification_01'
d=json.loads((report/'validation.json').read_text());assert d['passed'] and not d['partial']
out=r/'final_export_01';out.mkdir(exist_ok=False)
for p in report.iterdir():
    if p.is_file():shutil.copy2(p,out/p.name)
shutil.copy2(r/'preflight_visuals_02/validated_waypoints.png',out/'validated_waypoints.png')

summary={
    'result':'Validated 14 static poses, but residual PPO did not improve overall transition reliability over the reference.',
    'training_transitions':294912,'active_transitions':288000,'warmup_transitions':6912,
    'static_retention':'14/14 poses, full20 seconds',
    'last_policies':{
        '54911':{'support_successes':6,'physical_failures':3,'standing_sitting':False,'standing_leaning':False,'trained_stages':[4,5,6]},
        '54917':{'support_successes':9,'physical_failures':3,'standing_sitting':False,'standing_leaning':True,'trained_stages':[3,4,5,6]}},
    'reference':{'support_successes':9,'physical_failures':2,'standing_successes':0},
    'fixed':'All14 retain posture without physical failure;7 are already-supported starts.',
    'effort':'On common successful sitting starts5/6, last-policy whole-trial total squared torque is18.96%/19.11% lower than holding fixed posture. Against reference motion, the corresponding common sitting transitions3–6 improve only1.97%/1.58%. Most of the improvement over fixed posture comes from the reference movement.',
    'leaning_effort':'On each policy’s common successes, whole-trial squared torque is11.69%/28.62% higher than fixed posture. Against reference it changes+1.14%/−1.56%; case sets differ.',
    'reproducibility':'Seed54917 stage3 lean fails the2-environment GPU1 gate but passes the14-environment GPU0 final evaluation. Starts differ by up to11.27 micrometers in translation,17.70 microradians in joint position and0.000675 in normalized observation components. This does not establish the cause of the outcome change; robust repeatability is unproven. Within the final comparison every controller has bitwise-identical starts.',
    'perception':'100% availability in recorded audit samples before physical failure; no missing motion samples at10Hz; worst plane offset error2.104mm and normal error0.365degrees, rounded upward.',
    'next':'Use separate sitting and leaning curricula, so leaning cannot block sitting progress. Validate contact-safe weight transfer between poses across environment placements before a larger run; keep the same physical acceptance gates and compare against reference-only motion. Any further run requires a new recorded protocol; none is launched here.',
    'promotion':'No policy promoted.'}
for seed,x in summary['last_policies'].items():
    actual=d['evaluations'][f'eval_{seed}_last'];assert actual['successes']==x['support_successes'] and actual['physical_failures']==x['physical_failures']
assert d['total_completed_transitions']==summary['training_transitions']
(out/'research_summary.json').write_text(json.dumps(summary,indent=2))
page='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Two-surface PPO — outcome</title><style>body{font:17px/1.5 system-ui;max-width:1060px;margin:40px auto;padding:0 24px;color:#183238}h1{font-size:32px}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #d4dfdc;padding:10px;text-align:left}img{width:100%;margin:18px 0}.callout{background:#edf3f0;padding:18px;border-radius:8px}a{color:#246979}small{color:#53676b}</style>
<h1>Stable waypoints built; transition learning remains incomplete</h1>
<p>Seven poses for a 40 cm seat and seven for a 45° leaning surface passed independent 20-second retention tests. Two PPO seeds then learned bounded corrections around a fixed reference motion. They trained progressively from supported starts, stopping when leaning failed the scheduled support gates.</p>
<p class="callout"><b>294,912 training transitions completed.</b> Neither policy improved overall transition reliability over reference-only motion. One policy leaned successfully from standing in the final test; neither sat successfully from standing. No policy is promoted.</p>
<table><tr><th>Controller</th><th>Support success</th><th>Physical failures</th><th>From standing</th></tr><tr><td>Reference only</td><td>9/14</td><td>2/14</td><td>0/2</td></tr><tr><td>PPO54911 · last trained</td><td>6/14</td><td>3/14</td><td>0/2</td></tr><tr><td>PPO54917 · last trained</td><td>9/14</td><td>3/14</td><td>1/2 · leaning</td></tr></table>
<p>Fixed posture retained all14 starts without physical failure; only7 starts already met the support criterion. The last-passing PPO checkpoints scored6/14 and8/14, also with3 physical failures each. Both failed standing starts. Earlier starts, including standing, were evaluated but never reached by the training curriculum.</p>
<p><a href="motion_replay.html">Open the interactive motion replay</a> · <a href="report.html">Detailed comparison and audit</a></p><img src="stability.png" alt="Outcomes by controller, surface and starting pose">
<h2>Effort: separate the benefit of moving from the benefit of learning</h2>
<p>Across common successful seated starts, the last PPO policies used about19% less whole-trial squared torque than fixed posture. Against the reference motion, the saving was only1.6–2.0% over the common sitting transitions. The reference therefore accounts for most of the observed relief.</p>
<p>Leaning effort increased11.7% and28.6% versus fixed posture on each policy’s common successes. Against reference motion it changed+1.1% and−1.6%, on different matched case sets. There is no consistent broad effort improvement. These are ratios of mean total squared joint torque, not measured energy; all37 joints count.</p>
<img src="paired_effort.png" alt="Paired torque changes on common successful trials"><p><a href="paired_effort.csv">Full per-case whole-trial, final10-second and final5-second effort data</a></p>
<h2>What stopped progress</h2>
<p>Leaning gates failed the friction-margin criterion while the robot remained nearly stationary, in legal contact and without excessive slip. Early sitting tests began failing with impact/contact violations and later fell. Valid static poses did not guarantee a good movement between them.</p>
<p>Tracking remained available in all recorded samples before failure: worst observed plane-distance error was below2.104mm and normal error below0.365°. This points to movement and contact handling rather than loss of the surface estimate.</p>
<p><b>Repeatability caveat:</b> the same54917 checkpoint failed one leaning start in the2-environment training gate and passed it in the14-environment final test. The runs used different layouts and GPUs, with small startup differences (up to11.27µm translation and17.70µrad joint position). We cannot attribute the change to one cause. All final controller-versus-baseline starts were exactly equal, but the isolated standing-leaning success is not evidence of robust behavior.</p>
<h2>Evidence and next step</h2><p>Verification passed33 frozen source/bank hashes and137 preserved prior hashes, finite changed weights, optimizer continuity, exact budgets (288,000 active +6,912 warmup), correct loaded checkpoints, full20-second evaluations, constant fixed commands and bounded learned commands.</p>
<p>Next: separate the sitting and leaning curricula, then validate weight transfer between poses across placements before another larger run. Keep the physical acceptance criteria and the reference-only baseline. No extra training was launched after this frozen study.</p>
<img src="validated_waypoints.png" alt="Seven validated poses on each of two surfaces"><p><small>Two known rigid surfaces; no unseen-surface or disturbance test. Success requires final5-second continuous support, including at least0.05 bodyweight normal load. It is not proof of complete weight transfer, global minimum torque, arbitrary-surface capability, or hardware readiness.</small></p>'''
(out/'findings.html').write_text(page)

evidence=out/'evidence';evidence.mkdir()
shutil.copy2(study/'protocol.json',evidence/'protocol.json');shutil.copytree(study/'frozen',evidence/'frozen')
for seed in [54911,54917]:
    shutil.copy2(study/f'worker_{seed}.json',evidence/f'worker_{seed}.json')
    state=json.loads((study/f'worker_{seed}.json').read_text())
    for block in state['history']:
        source=study/block['train'];destination=evidence/block['train'];destination.mkdir()
        for name in ['initial.pt','final.pt','manifest.json','verification.json','progress.json','results.json','episodes.jsonl']:
            shutil.copy2(source/name,destination/name)
        shutil.copytree(study/block['evaluation'],evidence/block['evaluation'])
for label in d['evaluations']:shutil.copytree(study/label,evidence/label)
shutil.copytree(r/'selected_path_01',evidence/'selected_path_01')
shutil.copy2(r/'preflight_verification_01.json',evidence/'preflight_verification_01.json')
for name in ['summarize_two_surface_study.py','render_two_surface_replay.py','verify_two_surface_study.py']:
    shutil.copy2(Path(__file__).parent/name,out/name)
files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in out.rglob('*') if p.is_file()}
(out/'SHA256SUMS.json').write_text(json.dumps(files,indent=2))
print(json.dumps(dict(export=str(out),files=len(files),MiB=sum(p.stat().st_size for p in out.rglob('*') if p.is_file())/2**20)))
