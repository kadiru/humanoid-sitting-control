"""Saved-only preservation of602/603 exact local quotient obstruction."""
from pathlib import Path
import json,datetime,hashlib,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert not (D/'preparation_handoff_record_603.json').exists()
 checks=[]
 for name in ['prior_hashes.json','completed_probes_preservation_601.json','supplemental_preservation_601.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 names=['pressure_quotient_compile_exit_602.json','pressure_quotient_freeze_exit_602.json','pressure_quotient_preparation_exit_602.json','pressure_quotient_audit_exit_603.json','pressure_quotient_preservation_compile_exit_603.json']
 exits={n:read(D/n) for n in names};assert all(x['actual_exit']==0 for x in exits.values())
 for p,h in read(D/'pressure_quotient_launch_602.json')['source_input_sha256'].items():assert sha(p)==h,p
 folders=['pressure_quotient_preparation_602','pressure_quotient_audit_603']
 for folder,name in zip(folders,names[2:4]):
  for p,h in read(D/folder/'SHA256.json').items():assert sha(D/folder/p)==h,p
  ex=exits[name];assert ex['main_manifest_sha256']==sha(D/folder/'SHA256.json') and ex['completion_timing_sha256']==sha(D/folder/'completion_timing.json')
  assert read(D/folder/'completion_timing.json')['through_output_manifest_elapsed_s']<=240
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');scope=read(D/'pressure_quotient_scope_603.json');review=read(D/'pressure_quotient_local_verification_603.json')
 assert prep['passed'] and audit['passed'] and prep['exact_obstruction_found'] and audit['exact_obstruction_found'] and review['all_hashes_verified']
 assert prep['result']==audit['result']=='exact_local_pressure_quotient_rejected' and not prep['proposed_quotient_certified']
 ws=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):ws.append(a)
 assert not ws,ws
 s=read(D/'status.json');notes=(D/'handoff_prepared_603.txt').read_text()
 assert s['last_sealed_closure']==str(D/'closure_record_601.json') and not s['task_complete'] and not s['main_training_started'] and not s['fresh_holdout_started']
 for src,dst in [(D/'status.json',D/'status_before_pressure_quotient_603.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md',D/'handoff_before_pressure_quotient_603.md')]:
  assert not dst.exists();shutil.copy2(src,dst)
 mf=read(D/'completed_probes_preservation_601.json');mf.update(read(D/'supplemental_preservation_601.json'))
 for folder in folders:
  for p in (D/folder).iterdir():
   if p.is_file():mf[str(p)]=sha(p)
 for p in (R/'scripts').glob('*pressure*60[23].py'):mf[str(p)]=sha(p)
 for p in D.glob('pressure_quotient_*'):
  if p.is_file() and p.name not in ['pressure_quotient_evidence_preservation_603.json','pressure_quotient_preservation_603.log']:mf[str(p)]=sha(p)
 for n in ['handoff_prepared_603.txt','status_before_pressure_quotient_603.json','handoff_before_pressure_quotient_603.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'pressure_quotient_evidence_preservation_603.json';assert not manifest.exists();manifest.write_text(json.dumps(mf,indent=2))
 now=datetime.datetime.now(datetime.timezone.utc).isoformat()
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
 s.update(utc=now,updated_utc=now,phase='offline_pressure_quotient602_exact_obstruction_audited603',latest_handoff=str(D/'handoff_prepared_603.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_audit_passed=True,latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_603.json'),latest_preparation_handoff=str(D/'handoff_prepared_603.txt'),last_sealed_closure=str(D/'closure_record_601.json'),latest_closure=str(D/'closure_record_601.json'),latest_attempt=str(D/'balance_path_598'),latest_audit_status='603 exact local rank-obstruction certificate PASS; proposed pressure quotient REJECTED; no primal/physical evidence.599/600 remain last sealed result/report.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_round_new_exact_pressure_cofactor_products=6,latest_round_new_exact_pressure_row_products=39,latest_round_new_exact_pressure_audit_products=168,latest_round_new_exact_local_pressure_obstructions=1,latest_round_new_root_fixture_scope='No fixture/model/root; NEW exact certificate on immutable572 coefficients.',new_solver_exists=False,new_solver_scope='602 exact necessary-condition test rejected the proposed local output-preserving pressure quotient. No604 formulation/complete algorithm/budget/candidate.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope=scope['scope'],latest_new_path_scope='Actual472INITIALX/Y/J+490canonicalaux/exactdrops unchanged. Latest actual LP598 no witness. No new reduced system/point/lift/motion/effort.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text((D/'pressure_quotient_front_603.txt').read_text()+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_601.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=ws,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,completed_scientific_calculations_repeated=False,scientific_execution_recovery=False,scientific_execution_failure=False,read_only_authoring_failures='pressure_quotient_authoring_602.json',proposed_quotient_certified=False,new_reduced_system=False,new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,numerical_lp_calls=0,numerical_linear_system_solves=0,exact_inverses=0,model_calls=0,root_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,primal_points=0,analytic_dual_vectors_constructed=0,graph_point_lifts=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_603.json').write_text(json.dumps(record,indent=2))
 files=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_603.txt',manifest,D/'preparation_handoff_record_603.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'pressure_quotient_scope_603.json',D/'pressure_quotient_local_verification_603.json']
 (D/'final_preparation_manifest_603.json').write_text(json.dumps({str(p):sha(p) for p in files},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
