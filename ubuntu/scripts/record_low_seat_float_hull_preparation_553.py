"""Preserve552/553 finite-domain preparation only; sealed551 and actual path remain unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_553.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_551.json','supplemental_preservation_551.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'float_hull_launch_552.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for n in ['float_hull_compile_exit_552.json','float_hull_freeze_exit_552.json','float_hull_preparation_exit_552.json','float_hull_audit_exit_553.json','float_hull_index_exit_553.json','float_hull_preservation_compile_exit_553.json']:
  exits[n]=read(D/n);assert exits[n]['actual_exit']==0
 folders=['float_hull_preparation_552','float_hull_audit_553']
 for n in folders:
  for p,h in read(D/n/'SHA256.json').items():assert sha(D/n/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'float_hull_local_verification_553.json');scope=read(D/'float_hull_scope_553.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'] and scope['lp_calls']==0
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_float_hull_553.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_float_hull_553.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_551.json');mf.update(read(D/'supplemental_preservation_551.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*float_hull*552*.py','*float_hull*553*.py','*float_hull_preparation_553.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['float_hull_*.json','float_hull_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='float_hull_preservation_553.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_553.txt','status_before_float_hull_553.json','handoff_before_float_hull_553.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'float_hull_evidence_preservation_553.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_553.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_float_hull552_audit553_preparation_complete',latest_handoff=str(D/'handoff_prepared_553.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_553.json'),latest_preparation_handoff=str(D/'handoff_prepared_553.txt'),last_sealed_closure=str(D/'closure_record_551.json'),latest_closure=str(D/'closure_record_551.json'),latest_attempt=str(D/'box_path_548'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='553 independently verifies binary64 endpoint adjacency, removed unary implications and retained row/domain copies. No LP or actual state audit.549 sealed473 geometry/work inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_finite_domain_systems=2,latest_round_new_binary64_endpoint_encodings=scope['new_endpoint_encodings'],latest_round_new_strict_endpoint_changes=scope['new_strict_endpoint_changes'],latest_round_new_singleton_rows_absorbed=scope['new_singleton_rows_absorbed'],latest_round_new_adjacency_certificates=scope['new_adjacency_certificates'],latest_round_new_unary_conic_coefficients=scope['new_unary_conic_coefficients'],latest_round_new_root_fixture_scope='NEW finite-binary64 domain preparation and independent endpoint adjacency/unary implication audit ONLY. ZERO LP, small fixture, old endpoint/mixed-row proof, metric, Gram, model/root/AD/geometry/finite-work/finite-encoding/physics. No candidate coordinates altered.',
 new_solver_exists=False,new_solver_scope='552 finite-domain structural preparation only; no554 numerical solver/full-path wrapper/budget/source freeze/robot witness exists. Completed548 remains read-only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='All78791 columns retained. Exact binary64 feasible set unchanged by replacing singleton rows with saved-endpoint directed binary64 bounds. Real feasible set is an inward subset,not generally equal; tolerance sets/original dual certificates not equivalent. Source retained coefficients/cost/E unchanged. No candidate contraction or physical waiver.',latest_new_path_scope='ActualX/Y/J/canonical auxiliary/exact drops unchanged. No numerical witness, motion, effort or solver performance claim.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 original_failure=read(D/'float_hull_audit_original_exit_553.json');assert original_failure['actual_exit']==1
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_551.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,old_scientific_calculations_repeated=False,execution_recovery=True,metadata_prerequisite_recovery=True,completed_scientific_calculations_repeated=False,original_audit_failure=original_failure,recovery_scope='Audit first invocation exited1 at missing actual_exit metadata key before hashes/output/arithmetic. Preserved original metadata/log/exit; added actual_exit alias to observed producer exit0. Unchanged audit then executed all arithmetic once successfully.',new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_553.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_553.txt',manifest,D/'preparation_handoff_record_553.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'float_hull_scope_553.json',D/'float_hull_local_verification_553.json']
 (D/'final_preparation_manifest_553.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
