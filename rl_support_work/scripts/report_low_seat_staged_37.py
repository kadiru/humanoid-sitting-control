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
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'progress_report_37';read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 a=read(D/'staged_approach_36/independent_staged_audit_36.json');assert a['software_audit_passed'];OUT.mkdir(exist_ok=False)
 dirs={'Original parent':D/'clearance_damped_01/baseline','Best normal3s':D/'normal_landing3s_06/candidate','Rejected joint17':D/'joint_path_17/candidate','Rate headroom31':D/'tracking_headroom_31/candidate','Staged approach36':D/'staged_approach_36/candidate'}
 episodes={k:read(v/'results.json')['episodes'] for k,v in dirs.items()};summaries={k:summarize(v) for k,v in episodes.items()};candidate=episodes['Staged approach36'];pairs={};per_geometry=[]
 windows=['metrics_whole_trial','metrics_last10s','metrics_last5s'];keys=['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']
 for label,base in episodes.items():
  if label=='Staged approach36':continue
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
 geometry=read(dirs['Staged approach36']/'geometry.json');trace=np.load(dirs['Staged approach36']/'trace.npz');ticks=np.load(dirs['Staged approach36']/'every_tick.npz');sensors=np.load(dirs['Staged approach36']/'sensor_inputs.npz')['sensors'];m=BodyModel('runs/g1_kinematics_geometry_04/model.json');cal=read(D.parent/'contact_effort_repair_01/plane_calibration.json');diag=[]
 for i,r in enumerate(candidate):
  nominal=geometry[r['geometry_index']];bad=(~ticks['no_slip'][:,i])|(ticks['wrong'][:,i]>10)|(~ticks['legal'][:,i])|ticks['failure'][:,i]
  first=np.flatnonzero(bad);event=float(ticks['t'][first[0]]) if len(first) else None
  entry=dict(case=i,height_cm=nominal['height']*100,slope_deg=nominal['pitch_deg'],stable=r['success'],clean=clean(r),slip_ticks=r['slip_ticks'],contact_union=bool(r['failures']['forbidden'] or r['illegal_ticks']),physical_forbidden_body_contact=bool((ticks['wrong'][:,i]>10).any()),illegal_patch_ticks=r['illegal_ticks'],physical_failures=r['failures'],conservative_warning_ticks=r['traction_bad_ticks'],first_failure_or_slip_s=event)
  if event:
   choices=np.flatnonzero(trace['t']<event);at=int(choices[-1]);time=float(trace['t'][at]);q=trace['q'][at,i];frame=sensors[0,round(time*10),i,270:405];assert np.array_equal(q[:23],frame[:23]);g=frame[69:72];geom=trace['geometry'][at,i];fresh=bool(geom[11] and geom[12]<=.5)
   entry['pre_event_sensor_geometry']=dict(sample_time=time,valid=bool(geom[11]),age_s=float(geom[12]),fresh=fresh,height_error_mm=None,slope_error_deg=None)
   if fresh:
    plane=measured(m,q,g,geom,[trace['estimated_root_x'][at,i],0,0],cal);entry['pre_event_sensor_geometry'].update(height_error_mm=(plane[3]-nominal['height'])*1000,slope_error_deg=plane[4]-nominal['pitch_deg'])
   else:
    available=np.flatnonzero((trace['t']<event)&(trace['geometry'][:,i,11]>0)&(trace['geometry'][:,i,12]<=.5));entry['pre_event_sensor_geometry']['last_fresh_sample_s']=float(trace['t'][available[-1]]) if len(available) else None;entry['pre_event_sensor_geometry']['fit_not_attempted']='Missing/stale plane; no substituted baseline sensors or truth plane.'
  diag.append(entry)
 (OUT/'geometry_and_outcomes.json').write_text(json.dumps(diag,indent=2))
 labels=list(summaries);metrics=['stable','clean','slip','unwanted_contact','physical_failure','conservative_margin_cases'];print('summary fields',summaries,flush=True)
 #Use independently recomputed episode fields to avoid terminology ambiguity.
 counts={k:dict(stable=sum(r['success'] for r in v),clean=sum(clean(r) for r in v),slip=sum(r['slip_ticks']>0 for r in v),contact=sum(r['failures']['forbidden'] or r['illegal_ticks']>0 for r in v),physical=sum(any(r['failures'].values()) for r in v),warnings=sum(r['traction_bad_ticks']>0 for r in v)) for k,v in episodes.items()}
 fig,axs=plt.subplots(1,2,figsize=(15,5));x=np.arange(len(labels));width=.24
 for off,key,color in [(-1,'stable','#7d9cab'),(0,'clean','#197662'),(1,'slip','#c38b25')]:
  vals=[counts[k][key] for k in labels];axs[0].bar(x+off*width,vals,width,label=key,color=color)
  for xx,val in zip(x+off*width,vals):axs[0].text(xx,val+.2,str(val),ha='center',fontsize=9)
 for off,key,color in [(-.5,'contact','#a83838'),(.5,'physical','#814d76')]:
  vals=[counts[k][key] for k in labels];axs[1].bar(x+off*.3,vals,.3,label=key,color=color)
  for xx,val in zip(x+off*.3,vals):axs[1].text(xx,val+.2,str(val),ha='center',fontsize=9)
 for ax in axs:ax.set_xticks(x,labels,rotation=17,ha='right');ax.set_ylim(0,26);ax.grid(axis='y',alpha=.2);ax.legend();ax.set_ylabel('Episodes /24')
 fig.suptitle('Full20s development probe — contact and slip outcomes before effort\nFrozen59117; no early61111 pulse, no PPO or fresh holdout');fig.tight_layout();fig.savefig(OUT/'comparison.png',dpi=160);plt.close(fig)
 phase=np.load(dirs['Staged approach36']/'staged_ticks_36.npz')
 fig,axs=plt.subplots(2,4,figsize=(15,6),sharex=True)
 phase_rows=[]
 for col,i in enumerate([0,2,6,10]):
  time=phase['t'][:,i];elapsed=phase['elapsed_before'][:,i]
  axs[0,col].plot(time,elapsed,label='Path elapsed');axs[0,col].plot(time,np.clip(time-1,0,14),'--',label='Unpaused plan')
  fresh=(phase['geometry'][:,i,11]>0)&(phase['geometry'][:,i,12]<=.5)
  axs[1,col].step(time,fresh.astype(int),where='post',label='Fresh plane')
  axs[1,col].fill_between(time,0,phase['paused'][:,i].astype(int),step='post',alpha=.45,color='#be7e24',label='Descent paused')
  hits=np.flatnonzero(ticks['wrong'][:,i]>10);contact=float(ticks['t'][hits[0]]) if len(hits) else None
  if contact is not None:
   for ax in axs[:,col]:ax.axvline(contact,color='#b03838',ls=':',label='First forbidden contact')
  axs[0,col].set_title(f"{diag[i]['height_cm']:g}cm / +{diag[i]['slope_deg']:g}°")
  axs[1,col].set_xlabel('Episode time (s)');axs[1,col].set_ylim(-.1,1.1)
  for ax in axs[:,col]:ax.set_xlim(0,20);ax.grid(alpha=.2)
 axs[0,0].set_ylabel('Path elapsed (s)');axs[1,0].set_ylabel('Availability / pause')
 axs[0,0].legend(fontsize=8);axs[1,0].legend(fontsize=8);fig.suptitle('Recorded path progress and current measured-plane availability\nPhysics and full20s outcome clock continue during every pause');fig.tight_layout();fig.savefig(OUT/'phase_and_geometry.png',dpi=160);plt.close(fig)
 for i in a['selected_cases']:
  time=phase['t'][:,i];fresh=(phase['geometry'][:,i,11]>0)&(phase['geometry'][:,i,12]<=.5)
  phase_rows.append(dict(case=i,paused_ticks=int(phase['paused'][:,i].sum()),paused_seconds=float(phase['paused'][:,i].sum()*.02),final_path_elapsed_s=float(phase['elapsed_before'][-1,i]),fresh_fraction=float(fresh.mean()),clean=clean(candidate[i]),stable=candidate[i]['success']))
 (OUT/'phase_and_geometry.json').write_text(json.dumps(phase_rows,indent=2))
 lost_stable=[i for i,(b,c) in enumerate(zip(episodes['Best normal3s'],candidate)) if b['success'] and not c['success']]
 files={'independent_audit.json':D/'staged_approach_36/independent_staged_audit_36.json','revision.json':D/'staged_approach_36/revision.json','sensor_control_audit.json':dirs['Staged approach36']/'sensor_control_audit.json','support_posture33.json':D/'support_posture_screen_33/results.json','expanded_posture34.json':D/'support_posture_envelope_34/results.json','staged_timing35.json':D/'staged_headroom_screen_35/results.json','actual_input_preparation36.json':D/'staged_preparation_36/preparation.json','source_clock_checks.json':D/'staged_preparation_36/source_clock_checks.json','prior_preservation.json':D/'prior_recheck_during_staged_36.json'}
 for name,path in files.items():shutil.copy2(path,OUT/name)
 visibility=read(D/'visibility_diagnosis_38/results.json')
 shutil.copy2(D/'visibility_diagnosis_38/results.json',OUT/'camera_visibility.json');shutil.copy2(D/'visibility_diagnosis_38/visibility.png',OUT/'camera_visibility.png')
 summary=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),task_complete=False,repair_complete=False,ppo_started=False,fresh_holdout_started=False,new_physical_episodes=24,active=24000,initial_settling_excluded=600,optimizer_updates=0,counts=counts,lost_best_probe_clean=a['lost_best_clean_cases'],gained_best_probe_clean=a['gained_best_clean_cases'],remaining_target_cases=a['remaining_target_cases'],added_physical_cases=a['added_physical_cases'],software_audit_passed=a['software_audit_passed'],unselected_full20_exact=a['exact_preservation_gate_passed'],geometry_pause_audit=a['geometry_pause_audit'],best_full72_unchanged=dict(clean=64,stable=70,slip=8,housing=2,original_retained=60,warnings=50,strict_clean=22),source_hashes={str(p):sha(p) for p in files.values()})
 summary.update(lost_best_probe_stable_cases=lost_stable,targeted_final_seat_support={str(i):float(trace['support'][-1,i]) for i in a['selected_cases']},visibility_diagnostic_sha256=sha(D/'visibility_diagnosis_38/results.json'))
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2))
 table=''.join('<tr><td>'+html.escape(k)+'</td>'+''.join(f'<td>{v[c]}</td>' for c in ['stable','clean','slip','contact','physical','warnings'])+'</tr>' for k,v in counts.items())
 etable=''
 for label,v in pairs.items():
  pair=v['common_clean'];values=[pair['windows'][w]['total_squared_torque_Nm2']['percent_change'] for w in windows]
  etable+=f'<tr><td>{html.escape(label)}</td><td>{pair["n"]}</td>'+''.join(f'<td>{x:+.4f}%</td>' if x is not None else '<td>Unavailable</td>' for x in values)+'</tr>'
 rows=''.join(f'<tr><td>{r["case"]}</td><td>{r["height_cm"]:g} / +{r["slope_deg"]:g}°</td><td>{"CLEAN" if r["clean"] else "FAILURE"}</td><td>{r["slip_ticks"]}</td><td>{r["contact_union"]}</td><td>{r["illegal_patch_ticks"]}</td></tr>' for r in diag)
 target_ok=not a['remaining_target_cases'] and not a['lost_best_clean_cases'] and not a['added_physical_cases']
 conclusion='The targeted probe prerequisite passed; complete72-case current-policy retention is still required before PPO.' if target_ok else 'The staged probe did not meet the physical repair prerequisite. It is not admitted to retention or PPO.'
 body=f'''<!doctype html><meta charset="utf-8"><title>Staged low-seat approach</title><style>body{{font:17px/1.5 system-ui;max-width:1180px;margin:38px auto;padding:0 24px;color:#24303c}}h1{{font-size:32px}}h2{{margin-top:32px}}.status{{padding:20px;background:#fff1d9;border-left:5px solid #b97d20}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{border-bottom:1px solid #ddd;text-align:left;padding:9px}}img{{width:100%}}a{{color:#0866a6}}</style>
<h1>Low-seat repair: staged approach and measured-geometry guard</h1><p class="status"><b>{conclusion}</b> The best verified full-development result remains 64/72 clean, 70 stable, eight slipping trials and two housing-contact cases. All 64 established successes must be retained. No PPO, fresh holdout or policy promotion occurred.</p>
<h2>Full 20-second physical outcomes</h2><table><tr><th>Controller</th><th>Stable</th><th>Clean</th><th>Slip</th><th>Contact union</th><th>Physical failure</th><th>Margin warning</th></tr>{table}</table><p>Unwanted contact is forbidden-body OR illegal support-patch contact. Conservative margin warnings are separate. Gained best-probe clean cases: {a['gained_best_clean_cases']}; lost: {a['lost_best_clean_cases']}; remaining targeted failures: {a['remaining_target_cases']}; added physical failures: {a['added_physical_cases']}. This isolating probe uses frozen 59117 and omits the early 61111 pulse; unchanged mild-slope failures are not current-policy regressions.</p><img src="comparison.png" alt="Complete development outcome counts">
<h2>What this controller changed</h2><p>Eight measured low, steep cases begin a separately scoped approach at 1 s. The planned motion reaches a 20 mm hover, holds for 0.583 s, descends along the measured plane normal for 1.75 s, then settles before 15 s. It keeps the previously tested −0.35 rad arm offset and 0.9 rad pelvis target. A static model selects a foot-supported endpoint; it does not use the case index or hidden geometry. Nominal commands remain below 0.175 rad/s; the separately tested 0.20 rad/s applied cap operates from 1–15 s on selected rows, then returns to 0.15. Filter and feedback gains are unchanged.</p><p>Foot anchors use the actual encoder/IMU sample at planning time. If current plane measurements are invalid or older than 0.5 s, the reference, landing and load-controller clocks pause before or during normal descent. Physics, parent timing and the full 20 s outcome clock continue. No stored plane is presented as a fresh measurement. Frozen parent 59117 still acts at 11/15 s. All physical criteria, materials and fixed fingers are unchanged.</p>
<h2>Recorded progress and sensing</h2><img src="phase_and_geometry.png" alt="Recorded phase clock, plane availability and pauses"><p>The guard paused {a['geometry_pause_audit']['paused_env_ticks']} environment-ticks in total. All eight targeted trials lost usable geometry before normal descent, held for 7.34 s, and ended with zero seat support. The missing slip flags therefore do not demonstrate a repaired sit. Best-probe stable cases lost: {lost_stable}. Three right hip-pitch housing contacts occurred at 16.32–17.14 s while the path remained held and the absolute-time frozen parent was active. This timing is not proof that the parent alone caused the contacts. Per-case pause duration and final progress are in <a href="phase_and_geometry.json">phase evidence</a>. Missing geometry before failures is reported as unavailable in <a href="geometry_and_outcomes.json">the complete geometry audit</a>; no truth or baseline fit is substituted.</p>
<h2>Camera coverage diagnosis</h2><img src="camera_visibility.png" alt="Unoccluded camera-frustum coverage from recorded poses"><p>The nominal rear-camera transform matches 144 stored camera poses within 0.24 micrometres in position and 2.1e−7 in rotation-matrix components. A diagnostic projection of the known finite seat from the recorded poses shows its visible patch becoming too narrow for the existing plane-fit screen, even before accounting for occlusion. The code-defined downward camera, currently disabled, has a much larger projected patch. This is a visibility hypothesis, not an actual second-camera measurement or a validated repair: no camera was enabled or fed into control. <a href="camera_visibility.json">The full diagnosis</a> records assumptions and limitations. Do not lower the plane-fit conditioning checks to manufacture availability.</p>
<h2>Independent checks</h2><p>Actual 417 inputs, action memory, frozen parent, full rewards, zero optimizer updates, planner replay, exact starts and first 1 s, command bounds and fixed fingers passed the software audit. Every 20 ms command and geometry-pause decision was checked. Unselected full-trajectory exactness: {a['exact_preservation_gate_passed']}. Existing rendered-measurement coupling is not declared fixed; differences remain reported with unchanged tolerances. Software checks do not establish physical success.</p>
<h2>Effort on common complete clean successes</h2><table><tr><th>Baseline</th><th>Paired trials</th><th>Whole 20 s torque²</th><th>Final 10 s torque²</th><th>Final 5 s torque²</th></tr>{etable}</table><p>Positive means more effort. <a href="paired_effort.json">All paired metrics</a> include all 37 joints, squared and absolute torque, time-mean instantaneous joint RMS Nm, leg/upper contributions and strict subsets where both controllers had no conservative-margin warning. <a href="by_case_comparison.json">Per-case comparisons</a> exclude failed fragments. These are controller measurements, not new PPO gains.</p>
<h2>Why this path was tested</h2><p>Bounded posture searches 33/34 found a supported 7 mm nominal clearance on two easier target geometries but did not find it on the two harder ones, even in the separately documented expanded posture family. That is a limited local-search result, not an impossibility proof. The staged timing check 35 passed all four geometric paths at the proposed nominal rate, while the old 0.15 rad/s timing failed. All eight actual-input preparations passed the existing 3 mm clearance, 5 mm foot-support and 15 mm seat-patch model screens; minimum nominal housing clearance was about 3.589 mm. The unachieved 7 mm research reserve is explicitly disclosed. None of these screens is a dynamic safety certificate.</p>
<h2>Every episode</h2><table><tr><th>Case</th><th>Height / slope</th><th>Outcome</th><th>Slip ticks</th><th>Contact union</th><th>Illegal-patch ticks</th></tr>{rows}</table><p>All 24 full-duration episodes are included. Prior studies and completed rejected probes remain unchanged. The fresh reserved holdout remains closed. Reference-assisted simulation on nearby rigid slabs is not unassisted sitting, real-robot validation, or sitting/leaning anywhere. Leaning has not been broadened. Torque is an effort proxy, not electrical energy or a global optimum.</p>
<p><a href="summary.json">Summary</a> · <a href="revision.json">Frozen experiment revision</a> · <a href="independent_audit.json">Independent audit</a> · <a href="prior_preservation.json">Prior-file verification</a></p><small>{summary['utc']} — Interim development evidence; requested repair and PPO comparison are unfinished.</small>'''
 (OUT/'findings.html').write_text(body)
 (OUT/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in sorted(OUT.iterdir()) if p.is_file()},indent=2));print(json.dumps({k:v for k,v in summary.items() if k!='source_hashes'},indent=2))
if __name__=='__main__':main()
