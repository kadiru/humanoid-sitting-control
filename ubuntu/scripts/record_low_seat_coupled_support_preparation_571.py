"""Preserve570/571 preparation; last sealed569 and actual path unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_571.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_569.json','supplemental_preservation_569.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'coupled_support_launch_570.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for n in ['coupled_support_compile_exit_570.json','coupled_support_freeze_exit_570.json','coupled_support_preparation_exit_570.json','coupled_support_audit_exit_571.json','coupled_support_index_exit_571.json','coupled_support_preservation_compile_exit_571.json']:
  exits[n]=read(D/n);assert exits[n]['actual_exit']==0
 folders=['coupled_support_preparation_570','coupled_support_audit_571']
 for folder in folders:
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'coupled_support_local_verification_571.json');scope=read(D/'coupled_support_scope_571.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'] and scope['robot_lp_calls']==0
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 notes=(D/'handoff_prepared_571.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 for p,name in [(D/'status.json','status_before_coupled_support_571.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_coupled_support_571.md')]:
  assert not (D/name).exists();shutil.copy2(p,D/name)
 mf=read(D/'completed_probes_preservation_569.json');mf.update(read(D/'supplemental_preservation_569.json'))
 for folder in folders:
  for p in (D/folder).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*coupled_support*570*.py','*coupled_support*571*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['coupled_support_*.json','coupled_support_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='coupled_support_preservation_571.log':mf[str(p)]=sha(p)
 for name in ['handoff_prepared_571.txt','status_before_coupled_support_571.json','handoff_before_coupled_support_571.md']:mf[str(D/name)]=sha(D/name)
 manifest=D/'coupled_support_evidence_preservation_571.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();counts=scope['counts']
 s.update(utc=now,updated_utc=now,phase='offline_coupled_support570_audit571_preparation_complete',latest_handoff=str(D/'handoff_prepared_571.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_571.json'),latest_preparation_handoff=str(D/'handoff_prepared_571.txt'),last_sealed_closure=str(D/'closure_record_569.json'),latest_closure=str(D/'closure_record_569.json'),latest_attempt=str(D/'dual_path_566'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='571 independent exact coupled-rate conic reconstruction and retained-copy audit complete. No LP or actual-state audit.567 sealed473 geometry/work inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_coupled_support_evaluations=counts['new_positive_ratio_support_evaluations'],latest_round_new_coupled_support_certified_rows=counts['proved_removed_rows'],latest_round_new_coupled_support_negative_targets=counts['insufficient_targets'],latest_round_new_coupled_support_coefficient_products=counts['independent_positive_conic_coefficient_products'],latest_round_new_coupled_support_rhs_products=counts['independent_positive_conic_rhs_products'],latest_round_new_root_fixture_scope='NEW exact mixed-rate premise plus inherited546-box residual support certificates. Counted new positive ratio tests/conic products. ZERO numerical LP/linear solve/oldproof/fixture/model/root/AD/geometry/finite encoding/physics.',
 new_solver_exists=False,new_solver_scope='570 exact implication preparation only. No572 complete numerical construction, whole-path wrapper, source freeze or robot witness.566 completed read-only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='All78791 original variables/domain/cost/E unchanged. All selected mixed-rate and singleton premises retained; remove only exact nonnegative-slack targets. Real implication does not establish tolerance-feasible-set equality, a feasible point, numerical performance or physical progress.',latest_new_path_scope='Actual472 INITIAL X/Y/J plus490 canonical auxiliary/exact drops unchanged. No numerical witness, motion or effort result.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_569.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,completed_scientific_calculations_repeated=False,scientific_execution_recovery=False,authoring_read_failure=scope['authoring_read_failure'],new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,numerical_lp_calls=0,robot_lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_571.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_571.txt',manifest,D/'preparation_handoff_record_571.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'coupled_support_scope_571.json',D/'coupled_support_local_verification_571.json']
 (D/'final_preparation_manifest_571.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
