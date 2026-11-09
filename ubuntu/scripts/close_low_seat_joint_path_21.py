from pathlib import Path
import hashlib,json,datetime,shutil,subprocess
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def newjson(p,d):
 with p.open('x') as f:json.dump(d,f,indent=2);f.write('\n')
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
stamp=now.replace(':','').replace('+','_')
a=json.loads((D/'joint_path_17/independent_joint_path_audit_v4_17.json').read_text())
assert a['audit_completed'] and not a['passed'] and not a['physical_admission_passed'] and not a['exact_preservation_gate_passed']
assert a['candidate']['clean']==14 and a['candidate']['physical_failures']==8
assert not subprocess.run(['tmux','list-sessions','-F','#{session_name}'],capture_output=True,text=True).stdout.strip()
rev=json.loads((D/'joint_path_17/revision.json').read_text())
assert all(sha(p)==h for p,h in rev['sources'].items())
old=json.loads((D/'completed_probes_preservation_14.json').read_text())
bad=[p for p,h in old.items() if not Path(p).is_file() or sha(p)!=h]
assert not bad,bad
newjson(D/'completed_probes_recheck_21.json',{'utc':now,'prior_manifest':'completed_probes_preservation_14.json','checked':len(old),'passed':True,'mismatches':[],'original_probe_sources_checked':len(rev['sources'])})
newjson(D/'joint_path_audit_recovery_status_17.json',{'utc':now,'phase':'audit_completed_candidate_rejected','audit_completed':True,'overall_audit_passed':False,'physical_admission_passed':False,'exact_preservation_gate_passed':False,'input_action_reward_and_planner_checks_passed':True,'candidate':a['candidate'],'best_probe':a['best_probe'],'added_physical_cases':a['added_physical_cases'],'remaining_target_cases':a['remaining_target_cases'],'no_physics_rerun_for_audit':True,'original_failed_status_preserved':'status_joint_path_17.json','audit':'joint_path_17/independent_joint_path_audit_v4_17.json','recovery':'joint_path_audit_recovery_17/recovery.json','main_training_started':False,'fresh_holdout_started':False})
archive=D/'completed_source_archive_21';archive.mkdir()
source_paths=set(map(Path,rev['sources']))
for p in (R/'scripts').glob('*low_seat*.py'):
 if p.stem.endswith(tuple('_'+str(i) for i in range(15,22))):source_paths.add(p)
for p in sorted(source_paths):shutil.copy2(p,archive/p.name)
newmap=dict(old)
for p in D.iterdir():
 if any('_'+str(i) in p.name for i in range(15,22)) and p.name!='completed_probes_preservation_21.json':
  for f in ([p] if p.is_file() else p.rglob('*')):
   if f.is_file() and '__pycache__' not in str(f):newmap[str(f)]=sha(f)
newjson(D/'completed_probes_preservation_21.json',newmap)
hist=D/'documentation_history_01';hist.mkdir(exist_ok=True)
for p in [D/'status.json',R/'docs/LOW_SEAT_CONTACT_STATUS.md',R/'docs/UBUNTU_CODEX_HANDOFF.md']:
 shutil.copy2(p,hist/(p.name+'.before_joint_path_21.'+stamp))
status={'utc':now,'phase':'physical_repair_unresolved_latest_candidate_rejected','latest_candidate':'joint_path_17','latest_candidate_outcome':a['candidate'],'latest_independent_audit_passed':False,'latest_exact_preservation_gate_passed':False,'best_verified_controller':'current_retention_09','retained_original':60,'original_retention_target':60,'development_clean_verified':64,'development_trials':72,'development_stable_verified':70,'remaining_verified_slip_cases':8,'remaining_verified_housing_cases':2,'preserve_clean_conditions_next':64,'main_training_started':False,'fresh_holdout_started':False,'physical_jobs_running':False,'policy_promoted':False,'latest_report':'progress_report_21/review.html','latest_audit_status':'joint_path_audit_recovery_status_17.json','completed_evidence_manifest':'completed_probes_preservation_21.json','completed_evidence_count':len(newmap),'initial_prior_hash_check':'prior_recheck_during_joint_path_17.json','local_report_original_hashes_verified':18,'local_supplemental_hashes_verified':4,'local_report_visually_inspected':True,'scope':'Development physical repair remains unresolved. No full repair or PPO admission. Audit recovery preserves genuine failed preservation and physical gates.'}
(D/'status.json').write_text(json.dumps(status,indent=2)+'\n')
notes=f'''Latest verified development result — {now}\n\nThe fifteenth physical candidate, joint_path_17, is COMPLETE and REJECTED: 16/24 stable, 14 clean, 6 slipping and 8 forbidden hip-housing-contact/physical-failure cases. It added six housing failures, gained no clean cases and retained all 14 best-probe clean outcomes. Lower slip counts in newly failed contact episodes are not successes. The best verified full-development controller remains guarded retention09: all 60 original successes retained, 64/72 clean, 70 stable, 8 slipping and 2 housing-contact cases. Preserve all 64 established clean conditions. No PPO, fresh holdout or promotion; no physical workers remain.\n\nRevision17 explicitly began the selected six approaches at8s and ended15s with a nominal rate-feasible joint path; the deepest36cm/+20deg kept previous routing. The independent v4 audit completed but overall passed=false: actual417 inputs/actions, frozen parent, rewards, zero optimizer, 24000 active/600 initial-settling ticks, exact starts/first8s, planner replay and bounds passed, while unselected full20s exact preservation FAILED. Measured geometry first diverged8.2s, before unselected command/body changes after10s; cause remains unresolved. Maximum unselected body-pose component difference2.08mm and q0.00265rad. No tolerance or criterion was relaxed. Original status_joint_path_17.json records the first offline-audit termination; joint_path_audit_recovery_status_17.json explains the completed rejected result.\n\nAudit recovery preserved both erroneous reconstruction attempts: use originalfloat32 runtime inputs and exact frozen bridge inertial model, not newly recorded COM metadata. These corrections passed the unchanged2e-7 planner tolerance; they did not repair the real preservation or physical failures. No physics was rerun. Recorded mesh diagnosis shows new housing contacts14.60–14.70s; selected roots were12–15mm inward and3.4–4.6mm below plan around14.6s. Fresh geometry was available. Nominal clearance was not a dynamic certificate; 0.1s pose samples bracket0.02s events. Diagnose tracking/sensed clearance and unselected measurement isolation before another versioned bounded repair, without simulator-truth/replayed-sensor substitution.\n\nOffline screens15/16/18/19/20 did not run additional physics or learning. The deepest-seat endpoint/whole-path optimization still fails the model clearance/rate combination; even velocity matching requires0.150119rad/s>0.15. Do not round it into admission. All original8 revision17 source hashes and all{len(old)} protected prior-probe files rechecked intact. Initial24,242 prior-study hashes passed in prior_recheck_during_joint_path_17.json. completed_probes_preservation_21.json protects{len(newmap)} completed evidence files.\n\nprogress_report_21 copied to validated_low_seat_contact_01: all18 original SHA entries and4 supplemental review hashes verified, comparison/housing plots and review.html visually inspected. URL http://127.0.0.1:8807/progress_report_21/review.html. This is interim development/controller evidence, NOT new PPO/unseen results. Paired whole20/final10/final5 squared/absolute/RMS/leg/upper effort and strict subsets use common complete clean episodes only. The rejected candidate's tiny paired differences are not an overall gain. Best retention09 still has mixed effort: vs selected62111@8 old development, whole20 torque²+4.38%,final5+25.45% on20 common clean cases. Automation remains ACTIVE; physical repair and requested PPO comparison are unfinished.\n\nEarlier entries follow.\n\n'''
for p in [R/'docs/LOW_SEAT_CONTACT_STATUS.md',R/'docs/UBUNTU_CODEX_HANDOFF.md']:
 p.write_text(notes+p.read_text())
newjson(D/'closure_record_21.json',{'utc':now,'task_complete':False,'new_evidence_count':len(newmap),'old_evidence_rechecked':len(old),'original_source_hashes_rechecked':len(rev['sources']),'report_original_hashes':18,'review_hashes':4,'report_visual_review':True,'main_training':False,'fresh_holdout':False,'notes':notes})
print(json.dumps({'prior_probe_files_verified':len(old),'frozen_probe_sources_verified':len(rev['sources']),'completed_manifest_count':len(newmap),'status':status['phase'],'utc':now}))
