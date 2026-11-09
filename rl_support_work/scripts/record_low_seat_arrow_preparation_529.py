"""Preserve528/529 operator preparation; last SEALED527 and physical path unchanged."""
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
 assert not (D/'preparation_handoff_record_529.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_527.json','supplemental_preservation_527.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'arrow_operator_launch_528.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for name in ['arrow_operator_compile_exit_528.json','arrow_operator_freeze_exit_528.json','arrow_operator_preparation_exit_528.json','arrow_operator_audit_exit_529.json']:
  exits[name]=read(D/name);assert exits[name]['actual_exit']==0
 folders=['arrow_operator_preparation_528','arrow_operator_audit_529']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'arrow_operator_local_verification_529.json');scope=read(D/'arrow_operator_scope_529.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'];assert scope['LP_columns_unchanged']==56951 and not scope['LP_or_Schur_equation_solved']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_arrow_operator_529.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_arrow_operator_529.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_527.json');mf.update(read(D/'supplemental_preservation_527.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*arrow_operator*528*.py','*arrow_operator*529*.py','*arrow_preparation*529*.py','*arrow_529.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['arrow_operator_*.json','arrow_operator_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='arrow_operator_preservation_529.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_529.txt','status_before_arrow_operator_529.json','handoff_before_arrow_operator_529.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'arrow_operator_evidence_preservation_529.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_529.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_arrow_operator528_audit529_complete_no_new_LP',latest_handoff=str(D/'handoff_prepared_529.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_529.json'),latest_preparation_handoff=str(D/'preparation_handoff_record_529.json'),last_sealed_closure=str(D/'closure_record_527.json'),latest_closure=str(D/'closure_record_527.json'),latest_attempt=str(D/'torque_path_524'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='529 exact support/copy proof plus independent floating normal-operator residuals. No new actual-state audit;525 sealed473 inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_operator_partitions=2,latest_round_new_weighted_gram_cholesky_blocks=scope['total_new_weighted_gram_cholesky_blocks'],latest_round_new_local_operator_rhs_solves=scope['total_new_local_two_rhs_solves'],latest_round_new_algorithm_unused_operator_examples=2,latest_round_new_root_fixture_scope='ZERO model/root/AD/geometry/finite-work/finite-map/LP/physics. NEW weighted Gram operator examples use2730 local Cholesky blocks and5460 right-hand sides, NOT old mechanics L/K/Q inverses or fixtures.',
 new_solver_exists=False,new_solver_scope='No530 complete LP algorithm/budget/witness.528/529 derives lossless local/separator normal-operator factorization only.524 remains latest complete bounded execution.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='All522 numeric rows and coefficients copied unchanged into local/separator pieces.1365 local17blocks,23205local variables,33746separator coordinates per system. Full projected LP still56951columns. No Schur equation or LP solved.',latest_new_path_scope='No actualX/Y/J/canonical auxiliary/drop change. No speedup, feasibility or uniform-weight conditioning claim. Original LP/finite/physical gates remain mandatory.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_527.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,old_scientific_calculations_repeated=False,new_solver_exists=False,new_complete_path=False,model_calls=0,root_calls=0,lp_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_529.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_529.txt',manifest,D/'preparation_handoff_record_529.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'arrow_operator_scope_529.json',D/'arrow_operator_local_verification_529.json']
 (D/'final_preparation_manifest_529.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
