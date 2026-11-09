"""Preserve530/531 HSD subsolver preparation; last SEALED527 and physical path unchanged."""
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
 assert not (D/'preparation_handoff_record_531.json').exists();checks=[]
 for name in ['prior_hashes.json','arrow_operator_evidence_preservation_529.json','arrow_operator_supplemental_529.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'schur_hsd_launch_530.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for name in ['schur_hsd_compile_exit_530.json','schur_hsd_freeze_exit_530.json','schur_hsd_preparation_exit_530.json','schur_hsd_audit_exit_531.json']:
  exits[name]=read(D/name);assert exits[name]['actual_exit']==0
 folders=['schur_hsd_preparation_530','schur_hsd_audit_531']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'schur_hsd_local_verification_531.json');scope=read(D/'schur_hsd_scope_531.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified'];assert scope['new_small_LP_calls']==2 and scope['new_full_size_LP_calls']==0 and audit['small_tests_returned_certified_witnesses']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_schur_hsd_531.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_schur_hsd_531.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'arrow_operator_evidence_preservation_529.json');mf.update(read(D/'arrow_operator_supplemental_529.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*schur_hsd*530*.py','*schur_hsd*531*.py','*schur_preparation*531*.py','*schur_531.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['schur_hsd_*.json','schur_hsd_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='schur_hsd_preservation_531.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_531.txt','status_before_schur_hsd_531.json','handoff_before_schur_hsd_531.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'schur_hsd_evidence_preservation_531.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_531.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_schur_hsd530_audit531_complete_small_tests_only',latest_handoff=str(D/'handoff_prepared_531.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_531.json'),latest_preparation_handoff=str(D/'preparation_handoff_record_531.json'),last_sealed_closure=str(D/'closure_record_527.json'),latest_closure=str(D/'closure_record_527.json'),latest_attempt=str(D/'torque_path_524'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='531 exact dyadic input scaling and all253 small-LP iterations independently checked. No full-size LP or actual-state audit;525 sealed473 inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_offline_lp_calls=2,latest_round_new_small_synthetic_lp_calls=2,latest_round_new_full_size_lp_calls=0,latest_round_new_small_gram_cholesky_blocks=scope['small_test_statistics']['local_cholesky_blocks'],latest_round_new_small_local_rhs_solves=scope['small_test_statistics']['local_rhs_blocks'],latest_round_new_small_DR_iterations=scope['small_test_statistics']['DR_iterations'],latest_round_new_small_CG_iterations=scope['small_test_statistics']['CG_iterations'],latest_round_new_exact_dyadic_source_scalings=2,latest_round_new_root_fixture_scope='ZERO actualmodel/root/AD/geometry/finitework/full-size LP/fullfinite map/physics. TWO NEW three-variable synthetic LP tests,253 DR steps/253CG iterations;4new1x1Gramblocks,1526localRHSblocksolves. Old528 full-size factors/examples inherited/read-only, not repeated.',
 new_solver_exists=True,new_solver_scope='530 complete mathematical HSD/Schur LP subsolver validated on two small LPs only. No532 full-size child/trajectory wrapper/budget freeze or robot witness. Bounded-memory trace and immediate nonfinite-candidate guard required before full-size adapter launch.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='Two full saved522 representations scaled by exact positive dyadic factors; all finite bounds appended. Old529 partition inherited. Two NEW small numerical primal-dual LP witnesses independently certified; no full-size problem solved.',latest_new_path_scope='ActualX/Y/J/canonical auxiliary/drops unchanged. Synthetic witnesses are not robot candidates. No speed,conditioning,full-size convergence or physical feasibility claim. Full original/finite/physical gates remain.', main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_527.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,old_scientific_calculations_repeated=False,new_solver_exists=True,new_complete_path=False,model_calls=0,root_calls=0,lp_calls=2,full_size_lp_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_531.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_531.txt',manifest,D/'preparation_handoff_record_531.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'schur_hsd_scope_531.json',D/'schur_hsd_local_verification_531.json']
 (D/'final_preparation_manifest_531.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
