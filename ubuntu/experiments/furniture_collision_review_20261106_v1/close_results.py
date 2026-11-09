"""Bind current evidence and account failures; no simulation or solver calls."""
from pathlib import Path
import json,hashlib,datetime,subprocess
import numpy as np
ROOT=Path('/home/kadir/humanoid-sitting');EXP=ROOT/'experiments';OUT=ROOT/'outputs/furniture_collision_review_20261106_v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
construction=[]
for name in ['furniture_collision_repair_20261106_v4','furniture_collision_repair_20261106_v5','furniture_collision_repair_20261106_v6','furniture_collision_repair_20261106_v7','furniture_collision_repair_20261106_v8','furniture_stool_placement_20261106_v1','furniture_fresh_view_construction_20261106_v1']:
 p=EXP/name/('models' if 'fresh_view' in name else 'seat_models')/'receipt.json';r=read(p);rows=r['cases']
 construction.append(dict(source=str(p),sha256=sha(p),cold_cases=len(rows),model_admitted=sum(x['model_admitted'] for x in rows),total_seconds=sum(x['seconds'] for x in rows),physics_steps=0))
trials=[]
for p in sorted((ROOT/'outputs/furniture_camera_coverage_20261106_v1/calibration_demos').iterdir()):
 audit=read(p/'independent_audit.json');signals=read(p/'controller_signals.json');d=dict(np.load(p/'trajectory.npz'))
 hover=next(i for i,s in enumerate(signals) if s['phase']=='hover');fail=int(np.flatnonzero(d['failure'][:,0])[0]);invalid=next(i for i,s in enumerate(signals) if not s['measured_geometry_valid'])
 qd=abs(d['qd_pre'][:,0,:23]);pitch=np.array([abs(s['pitch']-s['pitch_reference']) for s in signals]);good=(qd.max(-1)<=.05)&(pitch<=np.deg2rad(.5));count=0;best=0
 for v in good[hover:fail]:count=count+1 if v else 0;best=max(best,count)
 initial=dict(np.load(p/'frame_0.npz'));world=initial['selected_points']@initial['root_rotation'].T+initial['root_position'];coverage=[]
 for i in [350,650]:
  z=dict(np.load(p/f'frame_{i}.npz'));P=(world-z['camera_position'])@z['camera_rotation'];uv=P[:,:2]/P[:,2,None]*np.array([z['K'][0,0],z['K'][1,1]])+z['K'][:2,2]
  inside=(P[:,2]>0)&(uv[:,0]>=0)&(uv[:,0]<128)&(uv[:,1]>=0)&(uv[:,1]<96);pix=np.floor(uv[inside]).astype(int);delta=z['depth'][pix[:,1],pix[:,0]]-P[inside,2]
  coverage.append(dict(t_s=i*.02,initial_patch_points=len(world),in_image=int(inside.sum()),visible=int((abs(delta)<=.015).sum()),occluded=int((delta<-.015).sum()),vertical_pixel_bounds=[float(uv[:,1].min()),float(uv[:,1].max())]))
 frames=read(p/'contact_mapping.json')['frames'];bad=next(f for f in frames if not f['mapping_valid'])
 trials.append(dict(case=p.name,physical_audit=audit,camera_depth_audit=read(p/'camera_depth_admission.json'),model_audit=read(p/'fresh_geometry_admission.json'),
   first_invalid_camera_control_s=invalid*.02,hover_start_s=hover*.02,normal_landing_started=any(s['phase']=='landing' for s in signals),
   longest_pre_failure_quiet_run_ignoring_camera_and_gyro=best,minimum_max_joint_speed_in_hover=float(qd[hover:fail].max(-1).min()),
   patch_visibility=coverage,first_mapping_invalid_physics_clock_s=bad['physics_clock_s'],
   primary_limit='Both ankle-roll joints kept moving before contact;25currentquiet gate never opened. Selectedpatch also leaves/occludesimage. Loadedslip occurs before later seam-mappingambiguity.',
   files={str(f):sha(f) for f in p.iterdir() if f.is_file()},source=str(p)))
prestarts=[]
for name in ['furniture_load_repair_20261106_v3','furniture_collision_collection_20261106_v2']:
 for p in (ROOT/'outputs'/name/'calibration_demos').glob('*'):
  if not (p/'results.json').exists():prestarts.append(dict(case=p.name,source=str(p),accounting='Interrupted during initialization; no active-control trajectory; exact warmupcount unavailable,0..100substeps',completed=False));continue
  r=read(p/'results.json');prestarts.append(dict(case=p.name,source=str(p),active_controls=r['active_transitions'],warmup_controls=r['warmup_transitions'],physics_substeps=4*r['simulator_control_steps'],completed=False))
gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,temperature.gpu,memory.used','--format=csv,noheader'],text=True)
workers=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader'],text=True)
result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),state='TWO_FULL_LCOUCH_TRIALS_FAILED_STOOL_UNRESOLVED_NO_PPO',
 human_authorization='Fix stool and L-couch from saved data; continue20demos/PPO/all72when4090free; unchanged action limits/success criteria.',
 actual_progress=['Corrected camera contact-plane association for the faceted L-couch; changed saved-data support construction, not the furniture mesh/material.',
 'Rejected translated old camera clouds after actual new resets exposed13-63mmfront-edge loss.',
 'Derived and tested54degree simulated rear-camera mounting angle instead of45degrees; initial depth-to-mesh and full collision/timing checks passed.',
 'Executed and independently audited two complete20s L-couchrollouts on4090only. Both failed. No successfuldemonstration claimed.',
 'Fixed NumPy audit JSON serialization; stopped repeated implementation failures; all aborted starts preserved.'],
 stool=dict(resolved=False,observed_surface_expanded_points=888,prior_selected_points=456,
   changed_model='Curved top represented by actual camera surface triangles; legal proximal-thigh support and45degreeobjectorientation considered.',
   outcome='Two actualposeconstructions failed unchanged balance/patchinset. A physically bounded closer-placement construction hit its existingLPcountcap; no retry or limitincrease. No stool physics.'),
 observations=dict(initial_camera_fix_not_full_path_fix=True,observer_all_depth_candidate_rejected='Saved-frame test recovered0of6lostpatchframes. Pointsactually leavecameraimage/areoccluded; no softwarePASSclaimed.',
   contact_mapping='Later mixedfacets/seams make aggregatecontactmappingambiguous. Zero validatedload must notbe described aszero physicalcontact. Earlier loadedslip independentlyfailsbothtrials.'),
 physical=dict(new_complete_rollouts=2,new_successful_demos=0,final_hold_s=[r['physical_audit']['final_hold_s'] for r in trials],new_PPO_updates=0,new_72case_checks=0,baseline_clean='64/72',accepted_old_failure_repairs=0,fresh_holdout='closed'),
 resource_accounting=dict(offline_constructions=construction,full_trials_active_controls=2000,full_trials_warmup_controls=50,full_trials_physics_substeps=8200,
   completed_pre_motion_reset_physics_substeps=sum(r.get('physics_substeps',0) for r in prestarts),interrupted_initialization_substeps='unknown0..100; not claimed zero',
   prior_generic_disk_guard_stop='no launches',disk_reclamation='Sparse allocation attempt preserved46historicalhashes but recoveredno usefulspace.',
   finite_demo_storage='256MiBperjobcap plus10GiBfree reserve; previous20GBgenericguard replaced by bounded batch accounting. Control/physicalgates unaffected.',
   original16howner_deadline_retained=True,recurring_automation=False,active_own_scientific_workers=0),
 pre_motion_starts=prestarts,trials=trials,gpu_snapshot=gpu,gpu_workers=workers,GPU4070='AIWorker untouched',
 next_work='Do not retry these paths or claimcollisionPASSasrepair. Construct a posture/controller that stabilizes bilateral ankle roll under unchanged gains/actionbounds and maintains fresh observation of an associated surface throughoutapproach; independent dynamic/visibility witness required. Stool also needs an admissible concrete collision/contact placement within existing resourcecaps. Then collect10clean sitting+10cleanleaning on5objects,validateforcecoverage/imitation,run4090PPOandall72andvideo. No runnablePPOqueueexists yet.')
(OUT/'execution_receipt.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(dict(state=result['state'],controls=2000,valid_demos=0,physics_substeps_known=8200+result['resource_accounting']['completed_pre_motion_reset_physics_substeps'],gpu=gpu),indent=2))
