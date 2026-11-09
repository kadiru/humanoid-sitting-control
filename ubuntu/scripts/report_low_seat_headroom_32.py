"""Report the completed bounded command-tracking experiment and model screens."""
from pathlib import Path
import json,hashlib,shutil,datetime,html
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_low_seat_contact_probe_v2 import summarize,clean
from humanoid_sitting.shared_frame import BodyModel
from low_seat_joint_path_controller_17 import measured
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'progress_report_32';read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=read(D/'tracking_headroom_31/independent_tracking_audit_v2_31.json');assert a['software_audit_passed'];OUT.mkdir(exist_ok=False)
 dirs={'Original parent':D/'clearance_damped_01/baseline','Best normal3s':D/'normal_landing3s_06/candidate','Rejected joint17':D/'joint_path_17/candidate','Rate headroom31':D/'tracking_headroom_31/candidate'}
 episodes={k:read(v/'results.json')['episodes'] for k,v in dirs.items()};summaries={k:summarize(v) for k,v in episodes.items()};candidate=episodes['Rate headroom31'];pairs={};per_geometry=[]
 windows=['metrics_whole_trial','metrics_last10s','metrics_last5s'];keys=['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']
 for label,base in episodes.items():
  if label=='Rate headroom31':continue
  pairs[label]={}
  for strict in [False,True]:
   ids=[i for i,(b,c) in enumerate(zip(base,candidate)) if clean(b) and clean(c) and (not strict or (b['traction_bad_ticks']==0 and c['traction_bad_ticks']==0))]
   stat={'cases':ids,'n':len(ids),'windows':{}}
   for window in windows:
    stat['windows'][window]={}
    for key in keys:
     b=np.array([base[i][window][key] for i in ids]);c=np.array([candidate[i][window][key] for i in ids]);stat['windows'][window][key]=dict(baseline=float(b.mean()) if len(b) else None,candidate=float(c.mean()) if len(c) else None,percent_change=float((c.mean()/b.mean()-1)*100) if len(b) else None)
   pairs[label]['strict_both' if strict else 'common_clean']=stat
  for b,c in zip(base,candidate):
   per_geometry.append(dict(baseline=label,case=c['case'],geometry_index=c['geometry_index'],baseline_clean=clean(b),candidate_clean=clean(c),gained=not clean(b) and clean(c),lost=clean(b) and not clean(c),metrics={w:{k:dict(baseline=b[w][k],candidate=c[w][k],percent_change=(c[w][k]/b[w][k]-1)*100) for k in keys} for w in windows} if clean(b) and clean(c) else None))
 (OUT/'paired_effort.json').write_text(json.dumps(pairs,indent=2));(OUT/'by_case_comparison.json').write_text(json.dumps(per_geometry,indent=2))
 geometry=read(dirs['Rate headroom31']/'geometry.json');trace=np.load(dirs['Rate headroom31']/'trace.npz');ticks=np.load(dirs['Rate headroom31']/'every_tick.npz');sensors=np.load(dirs['Rate headroom31']/'sensor_inputs.npz')['sensors'];m=BodyModel('runs/g1_kinematics_geometry_04/model.json');cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');diag=[]
 for i,r in enumerate(candidate):
  nominal=geometry[r['geometry_index']];bad=(~ticks['no_slip'][:,i])|(ticks['wrong'][:,i]>10)|(~ticks['legal'][:,i])|ticks['failure'][:,i]
  first=np.flatnonzero(bad);event=float(ticks['t'][first[0]]) if len(first) else None
  entry=dict(case=i,height_cm=nominal['height']*100,slope_deg=nominal['pitch_deg'],stable=r['success'],clean=clean(r),slip_ticks=r['slip_ticks'],forbidden_contact=r['failures']['forbidden'],illegal_patch_ticks=r['illegal_ticks'],physical_failures=r['failures'],conservative_warning_ticks=r['traction_bad_ticks'],first_failure_or_slip_s=event)
  if event:
   choices=np.flatnonzero(trace['t']<event);at=int(choices[-1]);time=float(trace['t'][at]);q=trace['q'][at,i];frame=sensors[0,round(time*10),i,270:405];assert np.array_equal(q[:23],frame[:23]);g=frame[69:72];plane=measured(m,q,g,trace['geometry'][at,i],[trace['estimated_root_x'][at,i],0,0],cal)
   entry['pre_event_sensor_geometry']=dict(sample_time=time,valid=bool(trace['geometry'][at,i,11]),age_s=float(trace['geometry'][at,i,12]),height_error_mm=(plane[3]-nominal['height'])*1000,slope_error_deg=plane[4]-nominal['pitch_deg'])
  diag.append(entry)
 (OUT/'geometry_and_outcomes.json').write_text(json.dumps(diag,indent=2))
 labels=list(summaries);metrics=['stable','clean','slip','unwanted_contact','physical_failure','conservative_margin_cases'];print('summary fields',summaries,flush=True)
 #Use independently recomputed episode fields to avoid terminology ambiguity.
 counts={k:dict(stable=sum(r['success'] for r in v),clean=sum(clean(r) for r in v),slip=sum(r['slip_ticks']>0 for r in v),contact=sum(r['failures']['forbidden'] or r['illegal_ticks']>0 for r in v),physical=sum(any(r['failures'].values()) for r in v),warnings=sum(r['traction_bad_ticks']>0 for r in v)) for k,v in episodes.items()}
 fig,axs=plt.subplots(1,2,figsize=(13,5));x=np.arange(len(labels));width=.24
 for off,key,color in [(-1,'stable','#7d9cab'),(0,'clean','#197662'),(1,'slip','#c38b25')]:
  vals=[counts[k][key] for k in labels];axs[0].bar(x+off*width,vals,width,label=key,color=color)
  for xx,val in zip(x+off*width,vals):axs[0].text(xx,val+.2,str(val),ha='center',fontsize=9)
 for off,key,color in [(-.5,'contact','#a83838'),(.5,'physical','#814d76')]:
  vals=[counts[k][key] for k in labels];axs[1].bar(x+off*.3,vals,.3,label=key,color=color)
  for xx,val in zip(x+off*.3,vals):axs[1].text(xx,val+.2,str(val),ha='center',fontsize=9)
 for ax in axs:ax.set_xticks(x,labels,rotation=17,ha='right');ax.set_ylim(0,26);ax.grid(axis='y',alpha=.2);ax.legend();ax.set_ylabel('Episodes /24')
 fig.suptitle('Full20s development probe — contact and slip outcomes before effort\nFrozen59117; no early61111 pulse, no PPO or fresh holdout');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
 fig,axs=plt.subplots(2,3,figsize=(13,6),sharex=True)
 for col,i in enumerate([0,6,10]):
  for label,style in [('Rejected joint17','--'),('Rate headroom31','-')]:
   tr=np.load(dirs[label]/'trace.npz');sel=(tr['t']>=8)&(tr['t']<=16);j=m.names.index('left_hip_pitch_joint');axs[0,col].plot(tr['t'][sel],(tr['reference'][:,i,j]-tr['command'][:,i,j])[sel],style,label=label)
   rate=np.diff(tr['command'][:,i,j])/np.diff(tr['t']);mask=(tr['t'][1:]>=8)&(tr['t'][1:]<=16);axs[1,col].plot(tr['t'][1:][mask],abs(rate[mask]),style)
  axs[0,col].set_title(f"{diag[i]['height_cm']:g}cm /+{diag[i]['slope_deg']:g}°");axs[1,col].axhline(.15,color='gray',ls=':',lw=1);axs[1,col].axhline(.20,color='gray',ls=':',lw=1);axs[1,col].set_xlabel('Episode time(s)')
  for row in [0,1]:axs[row,col].grid(alpha=.2)
 axs[0,0].set_ylabel('Reference−command(rad)');axs[1,0].set_ylabel('Sampled hip rate(rad/s)');axs[0,0].legend();fig.suptitle('Recorded tracking response;0.1s plot samples, separate every20ms cap audit');fig.tight_layout();fig.savefig(OUT/'tracking.png',dpi=160);plt.close(fig)
 events=[]
 for label in ['Rejected joint17','Rate headroom31']:
  pp=dirs[label];tr=np.load(pp/'trace.npz');tk=np.load(pp/'every_tick.npz');land=read(pp/'landing_controller_audit.json')
  for i in [0,1,6,7,10,11]:
   hit=np.flatnonzero(tk['wrong'][:,i]>10.);event=float(tk['t'][hit[0]]) if len(hit) else None
   if event is None:continue
   at=int(np.argmin(abs(tr['t']-event)));r=next((v for row in reversed(land) if row['tick']*.02<=event for v in row['states'] if v['case']==i),None)
   events.append(dict(controller=label,case=i,first_forbidden_contact_s=event,sampled_time_s=float(tr['t'][at]),hip_reference_command_gap_rad=float(tr['reference'][at,i,0]-tr['command'][at,i,0]),estimated_root_minus_target_m=(np.array(r['position'])-r['target']).tolist() if r else None))
 (OUT/'tracking_events.json').write_text(json.dumps(dict(scope='Sensor-based root estimator and0.1s command samples near0.02s events; no exact-event body-pose claim. Reduced tracking error does not certify contact clearance.',rows=events),indent=2))
 files={'independent_audit.json':D/'tracking_headroom_31/independent_tracking_audit_v2_31.json','revision.json':D/'tracking_headroom_31/revision.json','sensor_control_audit.json':dirs['Rate headroom31']/'sensor_control_audit.json','staged_model_screen28.json':D/'staged_approach_screen_28/results.json','early_anchor_screen29.json':D/'early_anchor_clearance_screen_29/results.json','recorded_tracking_diagnosis30.json':D/'command_tracking_diagnosis_30/results.json','audit_recovery.json':D/'tracking_headroom_audit_recovery_31/recovery.json'}
 for n,p in files.items():shutil.copy2(p,OUT/n)
 summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_complete=False,repair_complete=False,ppo_started=False,fresh_holdout_started=False,new_physical_episodes=24,active=24000,initial_settling_excluded=600,optimizer_updates=0,counts=counts,lost_best_probe_clean=a['lost_best_clean_cases'],gained_best_probe_clean=a['gained_best_clean_cases'],remaining_target_cases=a['remaining_target_cases'],added_physical_cases=a['added_physical_cases'],software_audit_passed=True,unselected_full20_exact=a['exact_preservation_gate_passed'],best_full72_unchanged=dict(clean=64,stable=70,slip=8,housing=2,original_retained=60,warnings=50,strict_clean=22),sources={str(p):sha(p) for p in files.values()})
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 table=''.join('<tr><td>'+html.escape(k)+'</td>'+''.join(f'<td>{v[c]}</td>' for c in ['stable','clean','slip','contact','physical','warnings'])+'</tr>' for k,v in counts.items())
 etable=''
 for label,v in pairs.items():
  p=v['common_clean'];vals=[p['windows'][w]['total_squared_torque_Nm2']['percent_change'] for w in windows];etable+=f'<tr><td>{html.escape(label)}</td><td>{p["n"]}</td>'+''.join(f'<td>{x:+.3f}%</td>' if x is not None else '<td>Unavailable</td>' for x in vals)+'</tr>'
 rows=''.join(f'<tr><td>{r["case"]}</td><td>{r["height_cm"]:g}/+{r["slope_deg"]:g}°</td><td>{"CLEAN" if r["clean"] else "FAILURE"}</td><td>{r["slip_ticks"]}</td><td>{r["forbidden_contact"]}</td><td>{r["illegal_patch_ticks"]}</td></tr>' for r in diag)
 body=f'''<!doctype html><meta charset="utf-8"><title>Low-seat tracking test</title><style>body{{font:17px/1.5 system-ui;max-width:1120px;margin:40px auto;padding:0 24px;color:#202832}}h1{{font-size:32px}}h2{{margin-top:32px}}.status{{background:#fff2da;padding:18px;border-left:5px solid #b66b00}}table{{border-collapse:collapse;width:100%;font-size:15px}}th,td{{padding:9px;border-bottom:1px solid #ddd;text-align:left}}img{{width:100%}}a{{color:#0866a6}}small{{color:#56616d}}</style><h1>Low-seat repair: bounded tracking test</h1><p class="status"><b>Physical repair and PPO prerequisite remain unresolved.</b> Best full72-case result remains64clean,70stable,8slipping,2housing contacts. All64 established clean conditions must be retained. This report covers one new24-case controller probe; no new PPO, fresh holdout or promotion.</p>
<h2>What changed</h2><p>One explicit pre-training control change gave the preserved8–15s joint path command-rate headroom:0.20rad/s instead of0.15 on six measured-selected approach rows, then0.15 from15s. The nominal path, filter0.1, feedback gains, frozen59117, materials and physical pass criteria stayed unchanged. The deepest36cm/+20° retained the best normal3s controller. This is a controller experiment, not a relaxed slip/contact test.</p>
<h2>Full-duration physical outcomes</h2><table><tr><th>Controller</th><th>Stable</th><th>Clean</th><th>Slip</th><th>Contact union</th><th>Physical failure</th><th>Margin warning</th></tr>{table}</table><p>Contact union includes forbidden-body OR illegal-support-patch cases. Margin-only warnings are separate. Lost best-probe clean cases:{a['lost_best_clean_cases']}; gained:{a['gained_best_clean_cases']}; unresolved targeted rows:{a['remaining_target_cases']}. This isolated probe omits the current early61111 pulse, so unchanged mild-slope failures are not current-policy regressions.</p><img src="comparison.png" alt="All full-duration outcome counts">
<h2>Tracking hypothesis and audit</h2><p>The earlier rejected path used near-cap commands in80–100% of sampled intervals before its six new contacts, with0.052–0.063rad reference-to-command lag. Sensor-based unloading was zero, contradicting premature unloading as that failure explanation. A larger command cap is a test of tracking reserve, not proof of cause.</p><p>The new run reduced hip reference-command gaps near the targeted contacts to about0.017–0.020rad, and estimated inward position error to about2–4mm. Contacts occurred later, around14.92–15.04s, but all six targeted rows still failed contact safety. Better tracking was insufficient to repair this path. See <a href="tracking_events.json">recorded event comparison</a>; these are sampled estimates, not exact-event truth poses.</p><img src="tracking.png" alt="Recorded command lag and rates"><p>Independent actual417-input/action/reward/parent/planner checks passed. Every20ms command is checked against its declared cap; fixed fingers and first8s exact starts passed. Unselected full20s exactness:{a['exact_preservation_gate_passed']}. Numerical differences remain explicitly reported; no tolerance was enlarged and old joint17 exact/physical gates remain failed. The offline auditor's original finger-name lookup was preserved and corrected to the model's numbered finger names; no simulation, physical threshold or controller was changed by that audit repair.</p>
<h2>Effort on common complete clean successes only</h2><table><tr><th>Baseline</th><th>Paired trials</th><th>Whole20 torque²</th><th>Final10 torque²</th><th>Final5 torque²</th></tr>{etable}</table><p>Positive means more effort. Detailed JSON includes absolute torque, time-mean instantaneous all37-joint RMS Nm, leg/upper components, per-case comparisons and both-controller strict-margin subsets. No failed fragments contribute savings. These are controller/existing-policy measurements, not PPO gains.</p>
<h2>Offline approach checks</h2><p>Two earlier starts(1s,4s),20mm hover,0.5s hold,normal descent and0.5s settling were screened with the unchanged0.15rad/s cap. All eight tested paths passed their geometry checks but failed timing before15s;0.12rad/s nominal headroom was slower. Earlier fixed-foot endpoint searches found best tested clearances7.08/3.70/6.95/5.66mm across the four difficult geometries. Only the first reached7mm research reserve. These bounded model checks are not dynamic safety or impossibility proofs.</p>
<h2>All24 episode outcomes</h2><table><tr><th>Case</th><th>Heightcm/slope</th><th>Result</th><th>Slip ticks</th><th>Forbidden contact</th><th>Illegal patch ticks</th></tr>{rows}</table><p>Measured geometry availability/errors before first failures are saved in <a href="geometry_and_outcomes.json">geometry and outcome audit</a>. Prior64/72 result still has50margin-warning cases and22strict clean. Reference-assisted rigid-slab simulation is not unassisted sitting, real-robot validation or sitting/leaning anywhere. Leaning has not broadened; torque is a proxy, not electrical energy or a global optimum.</p><p><a href="summary.json">Summary</a> · <a href="independent_audit.json">Independent audit</a> · <a href="paired_effort.json">Paired effort and strict subsets</a> · <a href="by_case_comparison.json">Per-case comparisons</a> · <a href="revision.json">Exact experiment revision</a></p><small>{summary['utc']} — Interim development results; requested physical repair and PPO comparison remain unfinished.</small>'''
 (OUT/'findings.html').write_text(body);(OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},indent=2));print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
