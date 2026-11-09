"""Preserve558/559 definition-lift preparation only; sealed557 and actual path remain unchanged."""
from pathlib import Path
import json,hashlib,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_559.json').exists();checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_557.json','supplemental_preservation_557.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'definition_launch_558.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for n in ['definition_compile_exit_558.json','definition_freeze_exit_558.json','definition_preparation_exit_558.json','definition_audit_exit_559.json','definition_index_exit_559.json','definition_preservation_compile_exit_559.json']:
  exits[n]=read(D/n);assert exits[n]['actual_exit']==0
 folders=['definition_preparation_558','definition_audit_559']
 for n in folders:
  for p,h in read(D/n/'SHA256.json').items():assert sha(D/n/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'definition_local_verification_559.json');scope=read(D/'definition_scope_559.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'] and scope['lp_calls']==0
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_definition_559.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_definition_559.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_557.json');mf.update(read(D/'supplemental_preservation_557.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*definition*558*.py','*definition*559*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['definition_*.json','definition_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='definition_preservation_559.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_559.txt','status_before_definition_559.json','handoff_before_definition_559.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'definition_evidence_preservation_559.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_559.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_definition558_audit559_preparation_complete',latest_handoff=str(D/'handoff_prepared_559.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_559.json'),latest_preparation_handoff=str(D/'handoff_prepared_559.txt'),last_sealed_closure=str(D/'closure_record_557.json'),latest_closure=str(D/'closure_record_557.json'),latest_attempt=str(D/'float_path_554'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='559 independently verifies NEW definition equalities, both signed source-row identities, dual-column cancellation and exact row/domain copies. No LP or actual state audit.555 sealed473 geometry/work inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_definition_variables=scope['new_definition_variables'],latest_round_new_definition_equalities=scope['new_definition_equalities'],latest_round_new_signed_source_row_identities=scope['new_signed_source_row_identities'],latest_round_new_dual_native_coefficient_identities=scope['new_dual_native_coefficient_identities'],latest_round_new_coordinate_dual_cancellations=scope['new_coordinate_dual_cancellations'],latest_round_net_stored_nonzeros_removed=scope['net_nonzeros_removed'],latest_round_new_root_fixture_scope='NEW exact local mechanics definition lifting and independent coefficient identities ONLY. ZERO LP/smallfixture/oldproof/metric/factor/model/root/AD/geometry/finite-encoding/physics. All original78791 coordinates retained; appended definitions are not physical authority.',
 new_solver_exists=False,new_solver_scope='558 exact extended formulation only; no560 numerical solver/full-path wrapper/budget/sourcefreeze/robot witness.554 completed read-only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='Exact real projection and objective equality to546 original-domain subset; all original coordinates retained, new free definition variables/equalities share opposite dense expressions. No universal finite-binary64 extension or speed/conditioning claim.552 inward finite bounds not used.',latest_new_path_scope='ActualX/Y/J/canonical auxiliary/exact drops unchanged. No numerical witness, motion, effort or solver performance claim.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_557.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,old_scientific_calculations_repeated=False,execution_recovery=False,completed_scientific_calculations_repeated=False,new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,lp_calls=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_559.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_559.txt',manifest,D/'preparation_handoff_record_559.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'definition_scope_559.json',D/'definition_local_verification_559.json']
 (D/'final_preparation_manifest_559.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
