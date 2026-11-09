"""Preserve exact capped-maximum538/539 preparation only; sealed535 and actual path unchanged."""
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
 assert not (D/'preparation_handoff_record_539.json').exists();checks=[]
 for name in ['prior_hashes.json','command_forest_evidence_preservation_537.json','command_forest_supplemental_537.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'capped_max_launch_538.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for name in ['capped_max_compile_exit_538.json','capped_max_freeze_exit_538.json','capped_max_preparation_exit_538.json','capped_max_audit_exit_539.json','capped_max_index_exit_539.json','capped_max_preservation_compile_exit_539.json']:
  exits[name]=read(D/name);assert exits[name]['actual_exit']==0
 folders=['capped_max_preparation_538','capped_max_audit_539']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'capped_max_local_verification_539.json');scope=read(D/'capped_max_scope_539.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified']
 assert scope['source_rows_verified']==1070623 and scope['native_coefficients_verified']==4209392 and scope['scalar_signs_verified']==406593
 assert scope['new_small_proximal_examples']==6 and scope['new_small_fenchel_dual_lift_examples']==3 and scope['lp_calls']==0 and not scope['new_complete_numerical_algorithm']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_capped_max_539.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_capped_max_539.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'command_forest_evidence_preservation_537.json');mf.update(read(D/'command_forest_supplemental_537.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*capped_max*538*.py','*capped_max*539*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['capped_max_*.json','capped_max_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='capped_max_preservation_539.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_539.txt','status_before_capped_max_539.json','handoff_before_capped_max_539.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'capped_max_evidence_preservation_539.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_539.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_capped_max538_audit539_preparation_complete',latest_handoff=str(D/'handoff_prepared_539.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_539.json'),latest_preparation_handoff=str(D/'handoff_prepared_539.txt'),last_sealed_closure=str(D/'closure_record_535.json'),latest_closure=str(D/'closure_record_535.json'),latest_attempt=str(D/'schur_path_532'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='539 independently verified all1070623 source rows/4209392 native coefficients/406593 scalar signs, six new exact proximal KKT examples and three Fenchel/dual-bound lifts. Preparation only; no actual-state audit.533 sealed473 inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_new_capped_max_systems=2,latest_round_new_capped_max_source_rows_verified=1070623,latest_round_new_capped_max_native_coefficients_verified=4209392,latest_round_new_capped_max_scalar_signs_verified=406593,latest_round_new_small_exact_proximal_examples=6,latest_round_new_small_fenchel_bound_lift_examples=3,latest_round_new_scalar_epigraph_eliminations=2,latest_round_new_root_fixture_scope='ZERO LP/full-size proximal update/model/root/AD/geometry/finite-work/Gram/linear solve/finite-map/physics. Two NEW exact-real capped-maximum source specializations; six NEW local exact proximal examples and three Fenchel/dual-bound identities. No prior scientific calculation repeated.',
 new_solver_exists=False,new_solver_scope='New exact functional formulation/proximal primitive only. No540 complete numerical algorithm, resource budget, whole-path wrapper or candidate.532 complete solver remains read-only.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='Exact min-over-epigraph reformulation of516 ORIGINAL representative systems with inherited517 real proof. Retains all78790 native directions including21840 torque auxiliaries; no522 projection/504 transform. All signed constraints remain hard or within capped maximum. One scalar LP coordinate removed per system, no criterion omitted.',latest_new_path_scope='ActualX/Y/J/canonical auxiliary/exact drops unchanged. No LP, mechanics, numerical robot candidate, motion or effort. Storage/function reformulation is not solver speed or physical feasibility.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_535.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,old_scientific_calculations_repeated=False,execution_recovery=False,new_functional_formulation=True,new_complete_numerical_algorithm=False,new_solver_exists=False,new_complete_path=False,model_calls=0,root_calls=0,lp_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_539.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_539.txt',manifest,D/'preparation_handoff_record_539.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'capped_max_scope_539.json',D/'capped_max_local_verification_539.json']
 (D/'final_preparation_manifest_539.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','new_complete_numerical_algorithm','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
