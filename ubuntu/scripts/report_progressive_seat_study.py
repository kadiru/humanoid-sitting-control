"""Export independently verified progressive sitting results."""
import argparse,csv,html,json,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from run_progressive_seat_study import DATA,read,sha
from report_seat_load_study import replay

def copy_evidence(source,target):
 target.mkdir()
 for name in ['results.json','geometry.json','manifest.json','startup.npz','startup_validation.json','trace.npz','first_failures.json','sensor_audit.json','initial.pt','parent_identity.json','action_contract.json','rolling_plan_audit.json','last_rolling_paths.npz','initial_settling.json']:
  if (source/name).exists():shutil.copy2(source/name,target/name)

def main():
 p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--verification',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();v=read(a.verification/'verification.json');assert v['passed'];out=a.output;out.mkdir(exist_ok=False)
 shutil.copytree(a.verification,out/'verification');shutil.copytree(a.study/'frozen',out/'frozen');shutil.copytree(DATA/'validation_export_01',out/'near_seat_validation')
 for name in ['protocol.json','holdout_barrier.json']:shutil.copy2(a.study/name,out/name)
 for name in ['admission.json','preflight_recoveries.json','reservation.json','approach_capture_parity.json']:shutil.copy2(DATA/name,out/name)
 old=read(DATA.parent/'seat_load_transfer_01/ppo_01/protocol.json');paths={f'development_{k}':Path(x) for k,x in old['development_baselines'].items() if k!='new_reference'};paths['development_new_reference']=DATA/'harness_standing_01';paths['near_reference']=DATA/'harness_near_01'
 for label in ['current_reference','current_56211','current_56217','new_reference']:paths['holdout_'+label]=a.study/('holdout_'+label)
 for seed in [57111,57117]:
  source=a.study/f'seed_{seed}';paths[f'development_{seed}']=source/'final_development_standing';paths[f'near_{seed}']=source/'final_development_near';paths[f'holdout_{seed}']=a.study/f'holdout_policy_{seed}'
  target=out/f'training_{seed}';target.mkdir();shutil.copy2(source/'curriculum.json',target/'curriculum.json')
  for row in v['curriculum'][str(seed)]:
   block=Path(row['checkpoint']).parent;dest=target/block.name;dest.mkdir()
   for name in ['initial.pt','critic_warmup.pt','final.pt','manifest.json','progress.json','verification.json','results.json','first_failures.json','initial_settling.json','action_contract.json']:
    if (block/name).exists():shutil.copy2(block/name,dest/name)
   gate=Path(row['evaluation']);copy_evidence(gate,target/gate.name)
 evidence=out/'evidence';evidence.mkdir()
 for label,path in paths.items():copy_evidence(path,evidence/label)
 records=[]
 for label,baselines in v['comparisons'].items():
  for baseline,c in baselines.items():
   for group,windows in c['grouped'].items():
    for window,metrics in windows.items():
     for metric,stats in metrics.items():records.append(dict(policy_label=label,baseline_label=baseline,group=group,window=window,metric=metric,**stats))
 with (out/'paired_metrics.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
 labels=['current_reference','current_56211','current_56217','new_reference','57111','57117'];names=['Current\nreference','Current\nPPO56211','Current\nPPO56217','Rolling\nreference','New\nPPO57111','New\nPPO57117']
 fig,axs=plt.subplots(1,2,figsize=(13,4.8))
 for ax,bank,n in zip(axs,['development','holdout'],[18,16]):
  counts=[v['evaluations'][f'{bank}_{x}']['successes'] for x in labels];bad=[v['evaluations'][f'{bank}_{x}']['physical_failures'] for x in labels];xx=np.arange(6)
  ax.bar(xx,counts,color=['#607d8b']*3+['#bb7452','#247c73','#559c82']);ax.set_ylim(0,n+3);ax.set_xticks(xx,names,fontsize=9);ax.set_ylabel('Full20-second stable successes');ax.set_title(f'{bank.title()} standing starts');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  for x,c,b in zip(xx,counts,bad):ax.text(x,c+.2,f'{c}/{n}\n{b} physical',ha='center',fontsize=9)
 fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
 fig,ax=plt.subplots(figsize=(10,4))
 for seed in [57111,57117]:
  records=v['curriculum'][str(seed)];total=0;x=[];y=[]
  for r in records:x.extend([total,total+r['updates']]);y.extend([r['stage'],r['stage']]);total+=r['updates']
  ax.plot(x,y,label=str(seed),marker='.',alpha=.8)
 ax.set_yticks([0,1,2,3,4,5],['Standing','Approach6s','Approach8s','Contact10s','Near11.5°','Near5.7°']);ax.set_xlabel('Completed PPO updates');ax.set_title('Actual training starts: advancement requires the frozen development gate');ax.grid(alpha=.2);ax.legend();fig.tight_layout();fig.savefig(out/'curriculum.png',dpi=160);plt.close(fig)
 summary=[];effort=[];sentences=[];fmt=lambda x:'—' if x is None else f'{x:+.1f}%'
 for label,name in zip(labels,names):
  e=v['evaluations']['holdout_'+label];summary.append(f"<tr><td>{name.replace(chr(10),' ')}</td><td>{e['successes']}/16</td><td>{e['physical_failures']}</td><td>{e.get('strict_successes','—')}</td></tr>")
 for seed in [57111,57117]:
  e=v['evaluations'][f'holdout_{seed}'];near=v['evaluations'][f'near_{seed}'];t=v['training'][str(seed)]
  sentences.append(f"Seed{seed}: reserved standing success {e['successes']}/16, {e['physical_failures']} physical failures; near-seat success {near['successes']}/18. Actual standing training: {t['standing_transitions']:,} physics transitions; final trained stage {t['last_trained_stage']}.")
  for baseline in ['current_reference','current_56211','current_56217','new_reference']:
   c=v['comparisons'][f'holdout_{seed}'][baseline]['grouped']['all'];whole=c['metrics_whole_trial']['total_squared_torque_Nm2'];last=c['metrics_last5s']['total_squared_torque_Nm2'];effort.append(f"<tr><td>{seed}</td><td>{baseline.replace('_',' ')}</td><td>{whole['common_successes']}</td><td>{fmt(whole['change_percent'])}</td><td>{fmt(last['change_percent'])}</td></tr>")
 (out/'findings.html').write_text(f'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Progressive sitting — verified comparison</title><style>body{{max-width:1120px;margin:40px auto;padding:0 22px;font:18px/1.5 system-ui;color:#20383d;background:#fafcfb}}h1{{line-height:1.2}}h2{{margin-top:34px}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:16px}}td,th{{padding:10px;border-bottom:1px solid #d1ded8;text-align:left}}.lead{{padding:20px;background:#e6f1ec;border-radius:10px}}.note{{color:#536b70}}a{{color:#176c70}}</style><h1>From near-seat weight transfer toward standing</h1><p class="lead">The contact-aware reference passed18/18 near-seated tests without unwanted contacts, slipping or friction-margin violations. PPO was trained separately with a fixed budget and gates that only permit earlier starts after successful development evaluations.</p><p>{' '.join(sentences)}</p><p><a href="motion_replay.html">Matched standing replay</a> · <a href="near_seat_validation/findings.html">Physical near-seat validation</a> · <a href="paired_metrics.csv">All paired effort metrics</a> · <a href="verification/verification.json">Independent verification</a></p><h2>Standing on reserved surfaces</h2><table><tr><th>Controller</th><th>Stable</th><th>Physical failures</th><th>Strict no-slip/traction</th></tr>{''.join(summary)}</table><p class="note">All trials start standing and last20s. Stable means no physical failure and continuously qualifying support in the final5s. Strict additionally requires zero slip, friction-margin or legal-contact violations across the entire trial. A dash means the older harness did not record this all-tick summary.</p><img src="comparison.png" alt="Standing success and physical failures"><h2>How far the curriculum progressed</h2><img src="curriculum.png" alt="Actual start stages during training"><p>Each run completed128 updates:8 critic-only plus120 actor-enabled.737,280 main physics transitions total include indexed-reset warmup. Initial settling contributes7,200 additional excluded ticks across16 training processes; separate optimizer smoke used5,760 transitions. Actor/critic and Adam state continued across blocks. No failed gate was bypassed.</p><h2>Total joint effort on common successes</h2><table><tr><th>New seed</th><th>Baseline</th><th>Common successes</th><th>Whole20s torque²</th><th>Final5s torque²</th></tr>{''.join(effort)}</table><p class="note">No savings claims use failed fragments. A reduction on a small surviving subset does not compensate for reduced reliability. CSV includes final10s, absolute torque, time-averaged instantaneous all-joint RMS Nm, leg/upper contributions, seat load and height/slope groups.</p><h2>Scope and limits</h2><p>The actor/critic use406 sensor/reference channels and bounded residuals on23 body joints;14 finger commands stay fixed. The pelvis path is planned from depth, encoders, IMU and body geometry. Ground truth is used for resets, rewards and evaluation only. Known horizontal placement and a motion reference remain explicit assistance.</p><p>Eight surfaces reserved before training are evaluated at two placements each. Both full-budget checkpoints finished before holdout evaluation, with no checkpoint selection or policy promotion. Two seeds and nearby rigid slabs do not establish arbitrary furniture generalization. Torque is a proxy, not electrical energy or a global optimum. Near-seat success must not be presented as learned standing success.</p></html>''')
 replay({'fixed_all':paths['holdout_current_reference'],'reference_all':paths['holdout_current_56211'],'eval_current_56217':paths['holdout_current_56217'],'eval_new_reference':paths['holdout_new_reference'],'eval_56811':paths['holdout_57111'],'eval_56817':paths['holdout_57117']},out/'motion_replay.html')
 p=out/'motion_replay.html';p.write_text(p.read_text().replace('New PPO 56811','New PPO 57111').replace('New PPO 56817','New PPO 57117').replace('New reference without PPO','Rolling reference without PPO'))
 shutil.copy2(__file__,out/'report_source.py');files=sorted(p for p in out.rglob('*') if p.is_file());(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in files},indent=2));print('\n'.join(sentences))
if __name__=='__main__':main()
