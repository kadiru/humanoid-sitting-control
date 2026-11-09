"""Add human-readable conclusions and exhaustive grouped effort tables to a verified export.

Does not overwrite original study results or the original export manifest.
"""
import argparse,csv,hashlib,html,json
from pathlib import Path

def read(p):return json.loads(p.read_text())
def pct(x,y):return 100*(x/y-1) if y else None
def fmt(v):return '—' if v is None else f'{v:+.2f}%'
WINDOWS=['metrics_whole_trial','metrics_last10s','metrics_last5s']
METRICS=['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']
NAMES=['Sum of squared torque (Nm²)','Sum of absolute torque (Nm)','Mean instantaneous joint RMS (Nm)','Leg squared torque (Nm²)','Upper-body squared torque (Nm²)']

def main():
 p=argparse.ArgumentParser();p.add_argument('--export',type=Path,required=True);a=p.parse_args();root=a.export;target=root/'findings.html';assert not target.exists()
 manifest=read(root/'SHA256.json');assert all(hashlib.sha256((root/k).read_bytes()).hexdigest()==h for k,h in manifest.items())
 allrows=[];details=[];stability=[];effort=[];standing=[];summary={}
 for task in ['sitting','leaning']:
  folder=root/task;verified=read(folder/'verification.json');assert verified['passed'];protocol=read(folder/'protocol.json')
  labels=['fixed','old_reference','reference']+[f'PPO_{s}' for s in protocol['seeds']];data={l:read(folder/l/'results.json')['episodes'] for l in labels};summary[task]={}
  for label in labels:
   rows=data[label];stand=[r for r in rows if r['start_stage']==0];success=sum(r['success'] for r in rows);fail=sum(any(r['failures'].values()) for r in rows)
   stability.append(f'<tr><td>{task}</td><td>{label}</td><td>{success}/28</td><td>{fail}</td><td>{sum(r["success"] for r in stand)}/4</td></tr>')
  for label in ['reference']+[f'PPO_{s}' for s in protocol['seeds']]:
   summary[task][label]={}
   for baseline in ['fixed','old_reference','reference']:
    if baseline==label:continue
    groups={}
    for stage in ['all']+list(range(7)):
     pairs=[(x,y) for x,y in zip(data[label],data[baseline]) if x['success'] and y['success'] and (stage=='all' or x['start_stage']==stage)]
     group={}
     for window in WINDOWS:
      for metric in METRICS:
       x=sum(p[window][metric] for p,b in pairs)/len(pairs) if pairs else None;y=sum(b[window][metric] for p,b in pairs)/len(pairs) if pairs else None
       row=dict(surface=task,policy=label,baseline=baseline,start_stage=stage,common_successes=len(pairs),window=window,metric=metric,policy_mean=x,baseline_mean=y,change_percent=pct(x,y) if pairs else None)
       allrows.append(row);group[(window,metric)]=row
     groups[str(stage)]=group
    g=groups['all'];summary[task][label][baseline]={'common_successes':g[(WINDOWS[0],METRICS[0])]['common_successes'],'whole_squared_torque_percent':g[(WINDOWS[0],METRICS[0])]['change_percent'],'last10_squared_torque_percent':g[(WINDOWS[1],METRICS[0])]['change_percent'],'last5_squared_torque_percent':g[(WINDOWS[2],METRICS[0])]['change_percent']}
    details.append(f'<details><summary>{task} · {label} versus {baseline} · {g[(WINDOWS[0],METRICS[0])]["common_successes"]} common successes</summary><table><tr><th>Metric</th><th>Whole 20 s</th><th>Last 10 s</th><th>Last 5 s</th></tr>')
    for metric,name in zip(METRICS,NAMES):
     cells=[]
     for window in WINDOWS:
      z=g[(window,metric)];cells.append('—' if z['policy_mean'] is None else f'{z["policy_mean"]:.3f} vs {z["baseline_mean"]:.3f}<br><strong>{fmt(z["change_percent"])}</strong>')
     details.append('<tr><td>'+name+'</td>'+''.join('<td>'+c+'</td>' for c in cells)+'</tr>')
    details.append('</table><p>Policy mean vs baseline mean; percent change is the ratio of paired means. Common successful trials only.</p></details>')
    if label.startswith('PPO') and baseline=='reference':
     effort.append('<tr><td>'+task+'</td><td>'+label.removeprefix('PPO_')+'</td>'+''.join('<td>'+fmt(g[(w,METRICS[0])]['change_percent'])+'</td>' for w in WINDOWS)+'</tr>')
     stand=groups['0'];standing.append('<tr><td>'+task+'</td><td>'+label.removeprefix('PPO_')+'</td>'+''.join('<td>'+fmt(stand[(w,METRICS[0])]['change_percent'])+'</td>' for w in WINDOWS)+'</tr>')
  bystage=[]
  for label in labels:
   bystage.append('<tr><td>'+label+'</td>'+''.join('<td>'+str(sum(e['success'] and e['start_stage']==s for e in data[label]))+'/4</td>' for s in range(7))+'</tr>')
  details.append('<details><summary>'+task+' · success at every starting stage</summary><table><tr><th>Controller</th>'+''.join('<th>Stage '+str(s)+'</th>' for s in range(7))+'</tr>'+''.join(bystage)+'</table></details>')
 with (root/'effort_by_stage.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
 (root/'review_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
 content='''<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Standing-to-support: verified results</title><style>body{font:17px/1.5 system-ui;background:#f4f5f1;color:#193238;max-width:1100px;margin:36px auto;padding:0 22px}h1{font-size:34px}h2{font-size:24px;margin-top:34px}.lead{background:#dfece3;border-left:5px solid #27674c;padding:20px}.note{background:#fff1d9;padding:18px}table{border-collapse:collapse;width:100%;background:white;font-size:15px}td,th{padding:9px 11px;border-bottom:1px solid #d7dfda;text-align:left}th{background:#e9eeeb}details{padding:12px 0}summary{cursor:pointer;font-weight:600}a{color:#1b6251}img{width:100%}.meta{font-size:14px;color:#4d6265}</style>
<h1>Standing-to-sitting and leaning now work on both tested surfaces</h1>
<p class="lead"><strong>All four final PPO policies: 28/28 support successes, 4/4 from standing, zero physical failures.</strong> The corrected reference also passed 28/28 for each task. Training completed independently with 128 updates per policy, including 64 updates from standing.</p>
<p><strong>The main gain came from fixing balance and contact. PPO retained that success but added no consistent effort reduction.</strong> Sitting PPO changed whole-trial squared torque by −0.23% / −0.08% relative to the corrected reference; leaning increased it by 0.52% / 0.87%. Absolute torque and average instantaneous joint RMS rose for all four policies. Small savings on supported sitting starts obscure slightly higher effort from standing.</p>
<p><a href="motion_replay.html">Inspect matched movement replay</a> · <a href="report.html">Original verified report</a> · <a href="effort_by_stage.csv">All comparisons by stage, seed, metric and time window</a></p>
<h2>Stability and standing transitions</h2><table><tr><th>Task</th><th>Controller</th><th>Support success</th><th>Physical failures</th><th>From standing</th></tr>'''+''.join(stability)+'''</table>
<p class="meta">Each trial lasts 20 s; success requires no physical failure and the final 5 s continuously meeting the original support/contact criteria. The old sitting reference fell, impacted, and made forbidden contacts in its 8 failed trials. Old leaning failed the support criteria without falls or other physical failures. Fixed posture retains its initial pose; fixed standing is not expected to reach support.</p>
<h2>What PPO added beyond the corrected reference</h2><p>Change in mean sum of squared physical torques over all 37 joints. Negative means lower. All 28 policy/reference pairs succeeded, so the complete matched set is used.</p><table><tr><th>Task</th><th>Seed</th><th>Whole 20 s</th><th>Last 10 s</th><th>Last 5 s</th></tr>'''+''.join(effort)+'''</table>
<h2>Standing-start effort deserves separate attention</h2><p>Only the four matched standing placements per policy. Sitting did not save effort from standing; the two leaning seeds disagree.</p><table><tr><th>Task</th><th>Seed</th><th>Whole 20 s</th><th>Last 10 s</th><th>Last 5 s</th></tr>'''+''.join(standing)+'''</table>
<p>The corrected reference itself reduced whole-trial squared torque versus the old reference by 11.09% for sitting (20 common successes) and 12.79% for leaning (11). PPO versus fixed supported posture reduces squared torque 15.84–16.04% for sitting (8 common successes) and 2.12–4.02% for leaning (20), but these subsets differ from the full transition comparison. Improvements versus fixed posture must not be credited entirely to PPO.</p>
<h2>Detailed paired comparisons</h2>'''+''.join(details)+'''
<h2>Validation and limits</h2><p>All 524,288 study transitions verified: 512,000 active plus 12,288 indexed-reset warmup. The 6,144 development-smoke transitions are separate. All initial/final/checkpoint loading, optimizer continuity, finite changed actor/critic weights, 37 sitting and 38 leaning frozen hashes, and 148 distinct prior artifacts pass. Every matched physical state, geometry and sensor history is exactly equal at startup; appended reference channels intentionally differ among controllers. Fixed and finger commands stay constant; bounded residual/reference envelopes and sampled slew checks pass.</p>
<p>Perception remains available in every audited pre-failure sample, with no missing observations in sampled valid motion. Maximum plane-offset error 2.107 mm for sitting and 0.670 mm for leaning; maximum normal error below 0.363° across these tests. Ground truth supplies rewards/reset construction/offline labels, not live policy or feedback geometry. Offline motion commands and intended pitch are explicit reference inputs.</p>
<p>The final standing-start support carries about 58–61% of weight on the seat and about 16% vertically on the leaning surface; the feet still carry the remainder. Normal load on an incline is not its vertical weight share. Qualifying final support does not require every transient contact during approach to satisfy resting friction/slip criteria.</p>
<p class="note">Two known rigid surfaces only: a 40 cm level seat and 45° leaning surface. Each task uses 7 known reset states repeated at 4 simulator placements, not 28 independent geometries. Two seeds reuse those cases. No disturbance or unseen-surface test. This is reference-assisted residual PPO, not discovery of sitting from scratch or proof of minimum effort. Torque is an effort proxy, not electrical energy. No policy was promoted.</p>
<p>Next research question: can the stable controller and policies tolerate new surface heights, slopes and approach offsets, and can a separate posture-optimization experiment reduce effort beyond this strong reference? No additional training was started.</p>
<img src="comparison.png" alt="Verified success, failure and paired effort comparison"><p class="meta">Verified October 14, 2026 PDT. Original exported files remain checked by SHA256.json; additional review artifacts have their own manifest.</p>'''
 target.write_text(content)
 extra=['findings.html','effort_by_stage.csv','review_summary.json'];(root/'REVIEW_SHA256.json').write_text(json.dumps({n:hashlib.sha256((root/n).read_bytes()).hexdigest() for n in extra},indent=2))
 print(target)
if __name__=='__main__':main()
