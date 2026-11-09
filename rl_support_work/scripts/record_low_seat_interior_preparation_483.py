"""Preserve new482/483 preparation ONLY; sealed closure481 stays unchanged."""
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
 assert not (D/'preparation_handoff_record_483.json').exists();checks=[]
 for n in ['prior_hashes.json','completed_probes_preservation_481.json','supplemental_preservation_481.json']:
  m=read(D/n)
  for p,h in m.items():assert sha(p)==h,p
  checks.append(dict(manifest=n,files=len(m),passed=True))
 folders=['interior_witness_preparation_482','interior_witness_audit_483']
 for n in folders:
  for p,h in read(D/n/'SHA256.json').items():assert sha(D/n/p)==h,p
 for p,h in read(D/'interior_preparation_launch_482.json')['sources'].items():assert sha(p)==h,p
 ex={n:read(D/n) for n in ['interior_preparation_exit_482.json','interior_audit_exit_483.json']}
 assert all(x['actual_exit']==0 for x in ex.values())
 prep=read(D/folders[0]/'preparation.json');audit=read(D/folders[1]/'independent_audit.json');local=read(D/'interior_local_verification_483.json')
 assert prep['passed'] and audit['passed'] and local['all_hashes_verified'] and not local['visual_review_claimed']
 workers=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=Path(__file__).name for x in a[1:]):workers.append(a)
 assert not workers,workers
 for p,n in [(D/'status.json','status_before_interior_483.json'),(R/'docs/LOW_SEAT_CONTACT_STATUS.md','handoff_before_interior_483.md')]:
  assert not (D/n).exists();shutil.copy2(p,D/n)
 mf=read(D/'completed_probes_preservation_481.json');mf.update(read(D/'supplemental_preservation_481.json'))
 for n in folders:
  for p in (D/n).rglob('*'):
   if p.is_file():mf[str(p)]=sha(p)
 for n in ['low_seat_interior_witness_482.py','prepare_low_seat_interior_witness_482.py','audit_low_seat_interior_witness_483.py','record_low_seat_interior_preparation_483.py']:
  mf[str(R/'scripts'/n)]=sha(R/'scripts'/n)
 for pattern in ['interior*.json','interior*.log']:
  for p in D.glob(pattern):
   if p.is_file():mf[str(p)]=sha(p)
 mf.pop(str(D/'interior_preservation_483.log'),None)
 for n in ['handoff_prepared_483.txt','status_before_interior_483.json','handoff_before_interior_483.md']:mf[str(D/n)]=sha(D/n)
 manifest=D/'interior_evidence_preservation_483.json';manifest.write_text(json.dumps(mf,indent=2))
 for p,h in mf.items():assert sha(p)==h,p
 now=datetime.datetime.now(datetime.timezone.utc).isoformat();notes=(D/'handoff_prepared_483.txt').read_text();status=read(D/'status.json')
 # All per-round activity counters reset; latest complete478 evidence remains.
 for k,v in list(status.items()):
  if k.startswith('latest_round_') and isinstance(v,(int,float)) and not isinstance(v,bool):status[k]=0
 status.update(utc=now,updated_utc=now,phase='offline_relative_interior_witness482_483_prepared_no_new_path',audit_jobs_running=False,model_jobs_running=False,physical_jobs_running=False,latest_active_run=None,latest_active_snapshot=None,latest_handoff=str(D/'handoff_prepared_483.txt'),latest_preparation_handoff=str(D/'preparation_handoff_record_483.json'),latest_preservation=str(manifest),completed_evidence_manifest=str(manifest),completed_evidence_count=len(mf),last_sealed_closure=str(D/'closure_record_481.json'),latest_closure=str(D/'closure_record_481.json'),latest_preparation=str(D/folders[0]),latest_preparation_audit=str(D/folders[1]/'independent_audit.json'),latest_preparation_audit_passed=True,latest_preparation_scope='New whole-path relative-interior witness LP assembly482 and independent483 arithmetic audit complete. All422107 rows/22653 exact intervals/certificates replayed. No LP/model/root/candidate evaluated; positive joint interior feasibility remains unknown.',latest_round_new_certificate_arithmetic_cases=5,latest_round_new_assembly_fixtures=2,latest_round_new_root_fixture_scope='Zero new model/root/AD/geometry/finite-work calls. Saved472INITIAL matrices only.',new_solver_exists=False,prospective_joint_solver_assembly_exists=True,new_executable_whole_path_wrapper_exists=False,new_constructed_model_path_exists=False,new_candidate_exists=False,main_training_started=False,ppo_started=False,fresh_holdout_started=False,holdout_started=False,policy_promoted=False,task_complete=False,repair_complete=False,automation_status='ACTIVE',notes=notes)
 (D/'status.json').write_text(json.dumps(status,indent=2));doc=R/'docs/LOW_SEAT_CONTACT_STATUS.md';doc.write_text(notes+'\n\nEarlier preserved entries follow.\n\n'+doc.read_text())
 record=dict(utc=now,scope=__doc__,prior_checks=checks,protected_files=len(mf),manifest=str(manifest),manifest_sha256=sha(manifest),workers=workers,actual_exits=ex,preparation=prep,independent_audit=audit,local_hash_review=local,last_sealed_closure=str(D/'closure_record_481.json'),new_closure=False,new_solver_exists=False,prospective_joint_solver_assembly_exists=True,new_executable_whole_path_wrapper_exists=False,new_candidate=False,model_calls=0,root_calls=0,geometry_calls=0,finite_work_columns=0,optimizer_calls=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0,best_clean='64/72',protected_clean=64,unresolved_cases=8,repair_complete=False,task_complete=False,automation_status='ACTIVE')
 (D/'preparation_handoff_record_483.json').write_text(json.dumps(record,indent=2));print(json.dumps({k:v for k,v in record.items() if k not in ['preparation','independent_audit','local_hash_review']},indent=2),flush=True)
if __name__=='__main__':main()
