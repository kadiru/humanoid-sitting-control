"""Preserve capped-maximum PDHG540/541 subsolver preparation only; sealed535 and actual path unchanged."""
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
 assert not (D/'preparation_handoff_record_541.json').exists();checks=[]
 for name in ['prior_hashes.json','capped_max_evidence_preservation_539.json','capped_max_supplemental_539.json']:
  m=read(D/name)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=name,files=len(m),passed=True))
 for p,h in read(D/'capped_pdhg_launch_540.json')['source_input_sha256'].items():assert sha(p)==h,p
 exits={}
 for name in ['capped_pdhg_compile_exit_540.json','capped_pdhg_freeze_exit_540.json','capped_pdhg_preparation_exit_540.json','capped_pdhg_audit_exit_541.json','capped_pdhg_index_exit_541.json','capped_pdhg_preservation_compile_exit_541.json']:
  exits[name]=read(D/name);assert exits[name]['actual_exit']==0
 folders=['capped_pdhg_preparation_540','capped_pdhg_audit_541']
 for name in folders:
  for p,h in read(D/name/'SHA256.json').items():assert sha(D/name/p)==h,p
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');review=read(D/'capped_pdhg_local_verification_541.json');scope=read(D/'capped_pdhg_scope_541.json')
 assert prep['passed'] and audit['passed'] and review['all_hashes_verified']
 assert scope['new_small_synthetic_LP_calls']==2 and scope['new_full_size_LP_calls']==0 and scope['small_PDHG_steps']==925
 assert scope['full_size_metric_rows_verified']==1070623 and scope['full_size_metric_columns_verified']==157580
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_capped_pdhg_541.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_capped_pdhg_541.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'capped_max_evidence_preservation_539.json');mf.update(read(D/'capped_max_supplemental_539.json'))
 for name in folders:
  for p in (D/name).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for pattern in ['*capped_pdhg*540*.py','*capped_pdhg*541*.py']:
  for p in (R/'scripts').glob(pattern):mf[str(p)]=sha(p)
 for pattern in ['capped_pdhg_*.json','capped_pdhg_*.log']:
  for p in D.glob(pattern):
   if p.is_file() and p.name!='capped_pdhg_preservation_541.log':mf[str(p)]=sha(p)
 for n in ['handoff_prepared_541.txt','status_before_capped_pdhg_541.json','handoff_before_capped_pdhg_541.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'capped_pdhg_evidence_preservation_541.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_541.txt').read_text();s=read(D/'status.json')
 for k in list(s):
  if k.startswith('latest_round_'):
   if isinstance(s[k],bool):s[k]=False
   elif isinstance(s[k],(int,float)):s[k]=0
   else:assert k=='latest_round_new_root_fixture_scope',k
 s.update(utc=now,updated_utc=now,phase='offline_capped_pdhg540_audit541_subsolver_preparation_complete',latest_handoff=str(D/'handoff_prepared_541.txt'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_preservation=str(manifest),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),latest_preparation_handoff_record=str(D/'preparation_handoff_record_541.json'),latest_preparation_handoff=str(D/'handoff_prepared_541.txt'),last_sealed_closure=str(D/'closure_record_535.json'),latest_closure=str(D/'closure_record_535.json'),latest_attempt=str(D/'schur_path_532'),latest_complete_construction=str(D/'interior_path_484'),latest_audit_status='541 independently proved two full-size exact diagonal step-norm bounds and checked925 saved SMALL PDHG transitions, proximal gaps and two exact reference optima. No independent LP/robot state audit.533 sealed473 geometry/work inheritance remains.',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,
 latest_round_offline_lp_calls=2,latest_round_new_small_synthetic_lp_calls=2,latest_round_new_full_size_lp_calls=0,latest_round_new_exact_pdhg_metric_systems=2,latest_round_new_small_PDHG_iterations=925,latest_round_new_small_weighted_proximal_evaluations=925,latest_round_new_small_scalar_reconstructions=2,latest_round_new_root_fixture_scope='TWO NEW small LP solves only,925 PDHG/proximal steps,2 scalar reconstructions;2 NEW full-size diagonal metric certificates. ZERO robot LP/fullsize iteration/linear-system solve/Gram/model/root/AD/geometry/finite-work/fullfiniteencoding/physics. No completed scientific calculation repeated.',
 new_solver_exists=True,new_solver_scope='540 complete capped-maximum native PDHG LP subsolver tested once on two NEW small LPs; no542 full-size child/whole-path wrapper/source freeze/independent actual-state auditor or robot witness exists.',new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,latest_preparation_scope='Exact dyadic step metrics for both538 native operators; Cauchy norm bound<=1/2. Source coefficients unchanged. Two new small LPs pass numerical KKT and independent saved-transition/known-optimum audit. Finite proximal screens are not a summable-error convergence theorem.',latest_new_path_scope='ActualX/Y/J/canonical auxiliary/exact drops unchanged. Two small numerical LP witnesses are not robot directions/targets. No robot solve/model/motion/effort or performance claim.',main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(s,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,preparation_only=True,new_closure=False,last_sealed_closure=str(D/'closure_record_535.json'),prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=exits,preparation=prep,independent_audit=audit,local_verification=review,saved_scope=scope,old_scientific_calculations_repeated=False,execution_recovery=False,new_functional_formulation_inherited=True,new_complete_numerical_algorithm=True,new_solver_exists=True,new_complete_path=False,model_calls=0,root_calls=0,lp_calls=2,full_size_lp_calls=0,finite_maps=0,physics_ticks=0,policy_updates=0,protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_541.json').write_text(json.dumps(record,indent=2))
 handoff=[D/'status.json',R/'docs/LOW_SEAT_CONTACT_PROTOCOL.md',doc,D/'handoff_prepared_541.txt',manifest,D/'preparation_handoff_record_541.json',D/folders[0]/'preparation.json',D/folders[1]/'independent_audit.json',D/'capped_pdhg_scope_541.json',D/'capped_pdhg_local_verification_541.json']
 (D/'final_preparation_manifest_541.json').write_text(json.dumps({str(p):sha(p) for p in handoff},indent=2))
 print(json.dumps({k:record[k] for k in ['utc','prior_checks','protected_files','manifest','manifest_sha256','workers','new_closure','new_complete_numerical_algorithm','task_complete','automation_status']},indent=2),flush=True)
if __name__=='__main__':main()
