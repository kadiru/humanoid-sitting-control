"""Package completed diagnostics without changing physical results or protocols."""
from pathlib import Path
import json,hashlib,shutil,datetime,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'progress_report_27'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 a=read(D/'sensor_prefix_v4_24/independent_audit.json');s=read(D/'sensed_clearance_23/results.json');fixed=read(D/'clearance_reserve_screen_25/results.json');feet=read(D/'foot_placement_screen_26/results.json')
 assert a['source_hashes_unchanged'] and all(v['exact'] for group in a['replay_to_original'].values() for v in group.values())
 for row in a['records']:
  for field in ['q','gravity','body','K','camera_position','camera_orientation']:assert row['unselected'][field]['exact']
  assert all(x['refit_valid'] and x['baseline_refit_to_recorded']['exact'] and x['changed_refit_to_recorded']['exact'] for x in row['per_case'])
 OUT.mkdir(exist_ok=False)
 files={'sensor_audit.json':D/'sensor_prefix_v4_24/independent_audit.json','depth_comparison.png':D/'sensor_prefix_v4_24/depth_comparison.png','sensed_clearance.json':D/'sensed_clearance_23/results.json','sensed_clearance_curves.npz':D/'sensed_clearance_23/curves.npz','fixed_foot_screen.json':D/'clearance_reserve_screen_25/results.json','foot_placement_screen.json':D/'foot_placement_screen_26/results.json','diagnostic_protocol.json':D/'sensor_prefix_v4_24/protocol.json'}
 for name,p in files.items():shutil.copy2(p,OUT/name)
 for mode in ['baseline','changed']:shutil.copy2(D/'sensor_prefix_v4_24'/mode/'diagnostic_accounting.json',OUT/(mode+'_diagnostic_accounting.json'))
 curves=np.load(D/'sensed_clearance_23/curves.npz');fig,axs=plt.subplots(2,4,figsize=(15,7),sharex=True)
 for row,label in enumerate(['best_normal3s','rejected_joint_path']):
  for col,i in enumerate([0,2,6,10]):
   x=curves[f'{label}_{i}'];ax=axs[row,col];z=(x[:,0]>=10)&(x[:,0]<=15.1);ax.plot(x[z,0],x[z,1]*1000,label='Measured plane + encoders',lw=1.7);ax.plot(x[z,0],x[z,3]*1000,label='Recorded mesh / finite seat',lw=1.3)
   ax.axhline(5,color='#b26b00',ls='--',lw=.8);ax.axhline(0,color='#b72828',lw=.8)
   rr=next(r for r in s['rows'] if r['controller']==label and r['case']==i)
   if rr['first_forbidden_s'] is not None:ax.axvline(rr['first_forbidden_s'],color='#b72828',ls=':',label='First forbidden contact')
   ax.set_ylim(-6,45);ax.set_title(f"{rr['height_m']*100:g} cm / +{rr['slope_deg']:g}°");ax.grid(alpha=.2)
   if col==0:ax.set_ylabel(('Best normal 3s' if row==0 else 'Rejected joint path')+'\nClearance (mm)')
   if row==1:ax.set_xlabel('Episode time (s)')
 handles,labels=axs[1,1].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3);fig.suptitle('Recorded development episodes: sensor warning is available, but late\n5 mm line is diagnostic only; 0.1 s poses bracket 0.02 s contact events',fontsize=14);fig.tight_layout(rect=[0,.06,1,.91]);fig.savefig(OUT/'clearance_diagnosis.png',dpi=160);plt.close(fig)
 fixed_rows=[]
 for r in fixed['results']:
  f=r['best_tested_feasible'];g=next(v for v in feet['results'] if v['case']==r['case'])['best_tested_feasible']
  fixed_rows.append(dict(case=r['case'],height_cm=r['measured_height_m']*100,slope_deg=r['measured_slope_deg'],fixed_max_tested_mm=f['clearance_m']*1000 if f else None,proposed_shift_mm=g['proposed_common_foot_shift_m']*1000 if g else None,proposed_clearance_mm=g['clearance_m']*1000 if g else None))
 summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_complete=False,physical_repair_admitted=False,best_verified=dict(controller='current_retention_09',clean=64,trials=72,stable=70,slip=8,housing=2,original_successes_retained=60,conservative_warnings=50,strict_clean=22),new_full20s_repair_episodes=0,ppo_updates=0,fresh_holdout=False,
  sensor_diagnosis='Both8.4s captures exactly replay original traces. Unselected own state/camera transforms remain exact; rendered depth first differs at8.2s by up to3.8147µm. Independent plane fits exactly reproduce both recorded geometries. This locates the first divergence in rendered measurements; renderer internal mechanism is not proven. Original unselected20s exact-preservation failure remains false; no tolerance relaxed.',
  static_render_replay='Attempts22 are inconclusive: unavailable support estimates/stale attached-camera transforms after static pose writes. No physics or causal conclusion from them.',
  diagnostic_accounting=dict(completed_pair_active=20160,completed_pair_initial_settling=1200,prior_original_baseline_active_upper_bound=10080,prior_instrumented_baseline_active=10080,prior_v3_baseline_active=120,prior_attempt_initial_settling=1800,total_diagnostic_active_upper_bound=40440,total_initial_settling=3000,normal_control_tick_s=.02,integration_substep_s=.005),
  execution_recoveries='Preserved originals: wrong simulator-substep count; capture-write indentation; first audit started before changed prefix completed. No physical/control/reward/criteria change. Failed diagnostic work is counted separately; no failed work is called a full episode.',
  clearance=s['summary'],model_screens=fixed_rows,physical_conclusion='No new physical repair. All64 established clean cases must be retained and remaining8 slips/2 housing cases removed before PPO. Proposed foot endpoints do not define safe stepping or dragging commands.',
  effort='No new complete-episode effort comparison. Retention09 remains mixed: versus selected62111@8 on20 common clean old-development cases, whole20 torque²+4.38%, final5+25.45%. Its current61111 common-clean comparison is exact. These are prior controller-policy results, not new PPO gains.')
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 table=''.join(f"<tr><td>{r['height_cm']:.2f} / +{r['slope_deg']:.2f}°</td><td>{r['fixed_max_tested_mm']:.2f} mm</td><td>{('No feasible endpoint found' if r['proposed_shift_mm'] is None else f'{r["proposed_shift_mm"]:.1f} mm; clearance {r["proposed_clearance_mm"]:.2f} mm')}</td></tr>" for r in fixed_rows)
 body=f'''<!doctype html><meta charset="utf-8"><title>Low-seat repair — diagnostic progress</title><style>body{{font:17px/1.55 system-ui;max-width:1120px;margin:42px auto;padding:0 24px;color:#202832}}h1{{font-size:32px}}h2{{font-size:23px;margin-top:32px}}.status{{background:#fff2da;padding:18px;border-left:5px solid #b66b00}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}th,td{{padding:11px;text-align:left;border-bottom:1px solid #ddd}}small{{color:#4d5965}}a{{color:#0866a6}}</style>
<h1>Low-seat repair remains unresolved</h1><p class="status"><b>64/72 clean; 70 stable; 8 slipping; 2 housing-contact cases.</b><br>All60 original successes and4 added successes remain the retention target. These are the unchanged verified controller09 results. No new full20s repair trial, PPO, fresh holdout or policy promotion.</p>
<h2>First measurement divergence located</h2><p>{html.escape(summary['sensor_diagnosis'])}</p><img src="depth_comparison.png" alt="Matched raw depth and micrometre-scale differences"><p>At8.4s some depth validity edges also differ; no finite-depth difference exceeds1mm. Both independent18-case plane refits exactly match recorded estimates. The rejected joint path still adds six housing failures. Static pose-rendering attempts were inconclusive and remain preserved.</p>
<h2>Clearance can be sensed, but warning comes late</h2><p>The5mm diagnostic warning preceded all2 best-normal and8 rejected-path contacts by0.08–0.20s. It also flagged6 of14 clean cases in each controller. Sensor estimates differed from recorded plane-clearance labels by roughly2mm near contact; this is not a validated collision-prevention controller.</p><img src="clearance_diagnosis.png" alt="Measured and actual housing clearances before contacts">
<h2>Bounded model checks</h2><p>Three bounded local searches per geometry used recorded7s encoder/IMU/plane inputs and the frozen body model. Pitch capped0.9rad, foot-support margin5mm, pelvis patch margin15mm, fixed previously tested arm offset−0.35rad. The7mm research reserve is not a changed physical threshold. Fixed-foot searches did not reach it. A separate search asks whether hypothetical foot placement within±60mm could help.</p><table><tr><th>Measured height / slope</th><th>Best tested fixed-foot clearance</th><th>Proposed common foot shift</th></tr>{table}</table><p><b>No stepping trajectory or physical validation exists for these endpoints.</b> They cannot be applied by dragging feet. A bounded local search is not a proof of impossibility, and endpoint clearance is not a whole-path certificate.</p>
<h2>Effort and validation status</h2><p>{html.escape(summary['effort'])}</p><p>Conservative margin warnings remain separate:50/72 warning cases and22 strict-clean cases in verified09. No claim of friction capacity, unassisted sitting, real-robot validation, or sitting/leaning anywhere is supported. Leaning has not broadened.</p><p>Completed sensor pair:20,160 active control ticks +1,200 initial-settling ticks. Preserved failed attempts add at most20,280 active +1,800 settling ticks; each20ms control tick contains four5ms integration substeps. Zero optimizer updates. All fresh holdout surfaces remain closed.</p><p><a href="summary.json">Summary and recovery accounting</a> · <a href="sensor_audit.json">Independent raw-sensor audit</a> · <a href="sensed_clearance.json">All48 recorded episode diagnoses</a> · <a href="fixed_foot_screen.json">Fixed-foot search</a> · <a href="foot_placement_screen.json">Hypothetical foot-placement search</a></p><small>{summary['utc']} — Interim development diagnosis, not final PPO comparison.</small>'''
 (OUT/'findings.html').write_text(body)
 (OUT/'SHA256.json').write_text(json.dumps({str(p.relative_to(OUT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file()},indent=2)+'\n')
 print(json.dumps(dict(report=str(OUT),files=len(read(OUT/'SHA256.json')),screens=fixed_rows),indent=2))
if __name__=='__main__':main()
