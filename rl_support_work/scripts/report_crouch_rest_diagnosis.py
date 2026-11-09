"""Development-only diagnosis report, preserving all failed trials."""
import argparse,json,html,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
rows=[]
for f in sorted(a.root.glob('*/results.json')):
 d=json.loads(f.read_text());ep=d.get('episodes',[])
 if not ep:continue
 row=dict(probe=f.parent.name,trials=len(ep),stable=sum(x['success'] for x in ep),strict=sum(x['strict_transfer'] for x in ep),physical=sum(any(x['failures'].values()) for x in ep),slip=sum(x['slip_ticks']>0 for x in ep),traction=sum(x['traction_bad_ticks']>0 for x in ep),full20s=all(x['duration_s']==20 for x in ep))
 rows.append(row);dest=a.output/'evidence'/f.parent.name;dest.mkdir(parents=True)
 for fn in ['results.json','first_failures.json','rolling_plan_audit.json','manifest.json','startup_validation.json','geometry.json']:
  if (f.parent/fn).exists():shutil.copy2(f.parent/fn,dest/fn)
plt.rcParams.update({'font.size':10})
fig,ax=plt.subplots(figsize=(11,max(5,len(rows)*.43)));y=np.arange(len(rows));ax.barh(y+.23,[r['stable'] for r in rows],.23,label='Stable final hold');ax.barh(y,[r['strict'] for r in rows],.23,label='Clean whole movement');ax.barh(y-.23,[r['physical'] for r in rows],.23,label='Physical failures',color='#c94a46');ax.set_yticks(y,[r['probe'].replace('_standing10_01','').replace('_01','') for r in rows]);ax.set_xlim(0,18);ax.set_xticks(range(0,19,2));ax.set_xlabel('Trials out of 18 (nine development surfaces × two placements)');ax.set_title('Transfer diagnosis — reference probes, no new PPO');ax.legend(loc='upper center',bbox_to_anchor=(.5,-.12),ncol=3);ax.invert_yaxis();fig.tight_layout();fig.savefig(a.output/'comparison.png',dpi=150);plt.close(fig)
headers=['Probe','Trials','Stable','Clean whole movement','Physical failure','Slip cases','Traction-margin cases']
table='<tr>'+''.join('<th>'+s+'</th>' for s in headers)+'</tr>'
for r in rows:table+='<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>' for k in ['probe','trials','stable','strict','physical','slip','traction'])+'</tr>'
text='''<!doctype html><meta charset="utf-8"><title>Crouch-to-rest diagnosis</title><style>body{font:17px system-ui;max-width:1200px;margin:40px auto;padding:0 24px;line-height:1.5;color:#1e293b}table{border-collapse:collapse;width:100%;font-size:14px}th,td{padding:9px;border-bottom:1px solid #cbd5e1;text-align:left}img{width:100%}.note{padding:16px;background:#fff3cd;border-left:5px solid #c89016}</style><h1>Crouch-to-rest: physical validation in progress</h1><p class="note">These are development reference probes. No new PPO has started. A low-effort pose after a forbidden contact is not a successful resting motion.</p><p>The old handoff abruptly releases loaded motor offsets. Smoothing the commands removes that jump but does not prevent hip-housing contact and loss of grip. The old reference also has brief slip during landing despite its final stable-hold score. The new motion preserves the clean first ten seconds and must pass whole-movement checks before training.</p><img src="comparison.png" alt="Full-trial outcomes by probe"><table>'''+table+'''</table><p>Stable: no latched physical failure and final five seconds of qualifying support. Clean whole movement additionally requires zero active-tick slip, traction-margin or illegal-contact violations. All completed probes run twenty seconds; outcomes after the first failure cannot support effort-saving claims.</p><p>Truth trajectories and contact labels are used for evaluation only. Runtime planning uses depth geometry, encoders, IMU, reference placement and robot-model parameters. Nine known rigid surfaces and repeated placements do not establish generalization.</p>'''
(a.output/'findings.html').write_text(text);(a.output/'results.json').write_text(json.dumps(rows,indent=2));shutil.copy2(a.root/'prior_verification_01.json',a.output/'prior_verification.json');shutil.copy2(__file__,a.output/'report_source.py')
hashes={str(p.relative_to(a.output)):hashlib.sha256(p.read_bytes()).hexdigest() for p in a.output.rglob('*') if p.is_file()};(a.output/'SHA256.json').write_text(json.dumps(hashes,indent=2));print(a.output)
