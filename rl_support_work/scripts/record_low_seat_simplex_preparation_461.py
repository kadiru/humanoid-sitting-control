"""Preserve completed arithmetic preparation only; last sealed closure is459."""
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
 assert not (D/'preparation_handoff_record_461.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_459.json','supplemental_preservation_459.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 folders=['simplex_preparation_460','simplex_audit_v2_461','simplex_audit_input_recovery_461']
 for n in folders:
  for p,h in read(D/n/'SHA256.json').items():assert sha(D/n/p)==h,p
 for p,h in read(D/'simplex_preparation_launch_460.json')['source_input_sha256'].items():assert sha(p)==h,p
 ex=read(D/'simplex_preparation_exits_461.json');assert ex['preparation460_actual_exit']==ex['corrected_audit_v2_actual_exit']==0 and ex['original_audit461_actual_exit']==1
 prep=read(D/'simplex_preparation_460/preparation.json');audit=read(D/'simplex_audit_v2_461/independent_audit.json');assert prep['passed'] and audit['passed']
 local=read(D/'simplex_local_verification_461.json');assert local['all_hashes_verified'] and not local['visual_review_claimed']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_simplex_461.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_simplex_461.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_459.json');mf.update(read(D/'supplemental_preservation_459.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 sources=['low_seat_simplex_retraction_460.py','prepare_low_seat_simplex_460.py','audit_low_seat_simplex_461.py','audit_low_seat_simplex_v2_461.py','record_low_seat_simplex_preparation_461.py']
 for n in sources:mf[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 for n in ['simplex_preparation_launch_460.json','simplex_preparation_exits_461.json','simplex_preparation_460.log','simplex_audit_461.log','simplex_audit_v2_461.log','simplex_local_verification_461.json','handoff_prepared_461.txt','status_before_simplex_461.json','handoff_before_simplex_461.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'simplex_evidence_preservation_461.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_461.txt').read_text();status=read(D/'status.json')
 status.update(utc=now,updated_utc=now,phase='offline_simplex_preparation_460_461_complete_no_candidate',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_prepared_461.txt'),latest_preparation_handoff=str(D/'preparation_handoff_record_461.json'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_459.json'),latest_preparation=str(D/'simplex_preparation_460'),latest_preparation_audit=str(D/'simplex_audit_v2_461/independent_audit.json'),latest_preparation_audit_passed=True,latest_preparation_scope='New bounded-conservation chart460 and independent corrected461v2 complete. Arithmetic only; no new solver, path or model evaluation.',latest_round_constructed_model_records=0,latest_round_new_complete_trial_paths=0,latest_round_new_model_derivative_fixture_columns=0,latest_round_new_arithmetic_map_fixture_coordinates=10541,latest_round_new_root_fixture_states=0,latest_round_new_finite_work_columns=0,latest_round_inherited_geometry_records=0,latest_round_new_full20s_trials=0,latest_round_initial_settling_ticks=0,latest_round_new_active_ticks=0,latest_round_new_diagnostic_prefixes=0,latest_round_new_aborted_case_attempts=0,latest_round_optimizer_updates=0,latest_round_actual_forward_filter_ticks=0,latest_round_command_ticks_per_device=0,latest_round_offline_lp_calls=0,latest_round_offline_milp_calls=0,latest_round_offline_nlp_calls=0,latest_round_offline_nonlinear_iterations=0,latest_round_offline_nonlinear_trial_evaluations=0,latest_round_offline_accepted_steps=0,new_solver_exists=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,preparation_summary=prep,independent_audit_summary=audit,local_hash_review=local,last_sealed_closure=str(D/'closure_record_459.json'),new_closure=False,new_solver_exists=False,new_candidate=False,model_calls=0,root_calls=0,optimizer_calls=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_461.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['preparation_summary','independent_audit_summary','local_hash_review']},indent=2),flush=True)
if __name__=='__main__':main()
