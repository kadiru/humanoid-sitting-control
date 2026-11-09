"""Preserve564/565 dual-balance preparation only; sealed563 and actual path unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_565.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_v2_563.json','supplemental_preservation_563.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'dual_balance_launch_564.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for n in ['dual_balance_compile_exit_564.json','dual_balance_freeze_exit_564.json','dual_balance_preparation_exit_564.json','dual_balance_audit_exit_565.json','dual_balance_audit_v2_compile_exit_565.json','dual_balance_index_exit_565.json','dual_balance_preservation_compile_exit_565.json']:
  exits[n]=read(D/n);assert exits[n]['actual_exit']==0
 folders=['dual_balance_preparation_564','dual_balance_audit_565']
 for n in folders:
  for p,h in read(D/n/'SHA256.json').items():assert sha(D/n/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'dual_balance_local_verification_565.json');scope=read(D/'dual_balance_scope_565.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'] and scope['lp_calls']==0
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 # Validate all fields before any mutable alias writes.
 notes=(D/'handoff_prepared_565.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 for p,n in [(D/'status.json','status_before_dual_balance_565.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_dual_balance_565.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_v2_563.json');mf.update(read(D/'supplemental_preservation_563.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*dual_balance*564*.py','*dual_balance*565*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['dual_balance_*.json','dual_balance_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='dual_balance_preservation_565.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_565.txt','status_before_dual_balance_565.json','handoff_before_dual_balance_565.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'dual_balance_evidence_preservation_565.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 s.update(utc=now,updated_utc=now,phase='offline_dual_balance564_audit565_preparation_complete',latest_handoff=str(D/'handoff_prepared_565.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_565.json'),latest_preparation_handoff=str(D/'handoff_prepared_565.txt'),last_sealed_closure=str(D/'closure_record_563.json'),latest_closure=str(D/'closure_record_563.json'),latest_attempt=str(D/'definition_path_560'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='565 independently verifies NEW exact transposed stationarity/slack/gap identities and unchanged original domain. No LP or actual-state audit.561 sealed473 geometry/work inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_transpose_coefficient_copies=scope['new_transpose_coefficient_copies'],latest_round_new_stationarity_coefficient_identities=scope['new_stationarity_coefficient_identities'],latest_round_new_slack_coefficient_identities=scope['new_slack_coefficient_identities'],latest_round_new_scalar_bound_column_identities=scope['new_scalar_bound_column_identities'],latest_round_new_signed_gap_coordinate_identities=scope['new_signed_gap_coordinate_identities'],latest_round_new_signed_gap_multiplier_identities=scope['new_signed_gap_multiplier_identities'],latest_round_new_root_fixture_scope='NEW exact dual balance transpose and independent KKT coefficient identities only. ZERO LP/oldproof/fixture/metric/factor/model/root/AD/geometry/finite-encoding/physics. Original78791 coordinates now identified with equality-marginal slots; optimizer primal holds nonnegative inequality/scalar-bound multipliers.',
 new_solver_exists=False,new_solver_scope='564 exact dual formulation only; no566 numerical solver/full-path wrapper/budget/sourcefreeze/robot witness.560 completed read-only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='Original546 real domain retained. Exact dual balances and weak-duality/KKT-pair correspondence. No actual feasible primal/dual point, speed or conditioning claim.552 finite bounds and558 definition extension not used.',latest_new_path_scope='ActualX/Y/J/canonical auxiliary/exact drops unchanged. No numerical witness,motion,effort or solver performance result.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_563.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,old_scientific_calculations_repeated=False,execution_recovery=True,recovery_scope=audit['recovery_scope'],completed_scientific_calculations_repeated=False,new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_565.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_565.txt',manifest,D/'preparation_handoff_record_565.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'dual_balance_scope_565.json',D/'dual_balance_local_verification_565.json']
 (D/'final_preparation_manifest_565.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
