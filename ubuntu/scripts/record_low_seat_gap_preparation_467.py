"""Preserve free-gap preparation and local root validation; last sealed closure465 unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_467.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_465.json','supplemental_preservation_465.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['gap_coordinate_preparation_v3_466','gap_coordinate_audit_467','gap_input_recovery_466','gap_flat_index_recovery_466','gap_root_batch_recovery_467','gap_root_initialization_v2_467','gap_root_initialization_audit_v2_467']
 for n in folders:
  for p,h in read(D/n/'SHA256.json').items():assert sha(D/n/p)==h,p
 for name in ['gap_preparation_launch_v3_466.json','gap_root_initialization_launch_v2_467.json']:
  launch=read(D/name)
  for p,h in launch['sources'].items():assert sha(p)==h,p
 ex=read(D/'gap_preparation_exits_467.json')
 for k in ['preparation_v3_actual_exit','algebra_audit_actual_exit','supplementary_root_v2_actual_exit','supplementary_audit_actual_exit']:assert ex[k]==0,k
 for k in ['preparation_original_actual_exit','preparation_v2_actual_exit','supplementary_root_original_actual_exit']:assert ex[k]==1,k
 prep=read(D/'gap_coordinate_preparation_v3_466/preparation.json');audit=read(D/'gap_coordinate_audit_467/independent_audit.json');root=read(D/'gap_root_initialization_v2_467/result.json');ra=read(D/'gap_root_initialization_audit_v2_467/independent_audit.json');assert audit['passed'] and root['passed'] and ra['passed']
 assert root['newton_updates']==[1,1];assert not prep['physical_admission'];local=read(D/'gap_local_verification_467.json');assert local['all_hashes_verified'] and not local['visual_review_claimed']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_gap_467.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_gap_467.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_465.json');mf.update(read(D/'supplemental_preservation_465.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 sources=sorted(set((R/'scripts').glob('*gap*466.py'))|set((R/'scripts').glob('*gap*467.py')))
 for p in sources:mf[str(p)]=sha(p)
 for pattern in ['gap*466.json','gap*467.json','gap*466.log','gap*467.log']:
  for p in D.glob(pattern):
   if p.is_file():mf[str(p)]=sha(p)
 for n in ['handoff_prepared_467.txt','status_before_gap_467.json','handoff_before_gap_467.md']:mf[str(D/n)]=sha(D/n)
 # This writer's still-open log must be added only after actual exit, separately.
 mf.pop(str(D/'gap_preservation_467.log'),None)
 manifest=D/'gap_evidence_preservation_467.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_467.txt').read_text();status=read(D/'status.json')
 status.update(utc=now,updated_utc=now,phase='offline_free_gap_preparation_466_467_complete_no_candidate',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_prepared_467.txt'),latest_preparation_handoff=str(D/'preparation_handoff_record_467.json'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_465.json'),latest_preparation=str(D/'gap_coordinate_preparation_v3_466'),latest_preparation_audit=str(D/'gap_coordinate_audit_467/independent_audit.json'),latest_preparation_root_audit=str(D/'gap_root_initialization_audit_v2_467/independent_audit.json'),latest_preparation_audit_passed=True,latest_preparation_scope='New free-gap/drop-simplex coordinates and full saved-row algebra audited. Supplementary six-state two-sign unpredicted-start roots each performed one Newton update; no complete path or optimizer. Original predictor-initialized roots needed zero updates and are not standalone solver validation.',latest_round_constructed_model_records=0,latest_round_interpolated_model_records=0,latest_round_new_complete_trial_paths=0,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=0,latest_round_new_root_fixture_states=24,latest_round_new_root_fixture_scope='12 predictor-initialized signed states needed zero updates plus12 new unpredicted-start signed states needed one update each; failed batch attempt evaluated initial values/J before RHS error, no completed root. These are local fixtures, not complete paths.',latest_round_new_finite_work_columns=0,latest_round_inherited_geometry_records=0,latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=0,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_nonlinear_trial_evaluations=0,latest_round_offline_accepted_steps=0,new_solver_exists=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,preparation_summary=prep,independent_audit_summary=audit,supplementary_root_result=root,supplementary_root_audit=ra,local_hash_review=local,last_sealed_closure=str(D/'closure_record_465.json'),new_closure=False,new_solver_exists=False,new_candidate=False,complete_path_root_calls=0,local_fixture_scope=status['latest_round_new_root_fixture_scope'],new_signed_fixture_states=24,original_input_recovery_recomputed_saved_array_algebra=True,optimizer_calls=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_467.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['preparation_summary','independent_audit_summary','supplementary_root_result','supplementary_root_audit','local_hash_review']},indent=2),flush=True)
if __name__=='__main__':main()
