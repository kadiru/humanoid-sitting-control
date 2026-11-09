"""Postflight only: no simulation launch or automatic retry."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import datetime,json,os,subprocess,hashlib
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');S=D/'incoming_finalization_status_310.json'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def pipeline(names,require_exact=False):
 steps=[];env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 for i,n in enumerate(names):
  with (D/(n+'.log')).open('x') as log:r=subprocess.run([str(R/'IsaacLab/.venv/bin/python'),str(R/'scripts'/n)],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
  steps.append(dict(script=n,returncode=r.returncode))
  if r.returncode:return dict(passed=False,steps=steps)
  if require_exact and i==0 and not read(D/'incoming_audit_v2_305/independent_audit.json')['diagnostic_checks_passed']:return dict(passed=False,steps=steps,reason='Exact diagnostic replay gate failed; no diagnosis admitted')
 return dict(passed=True,steps=steps)
def main():
 assert not S.exists()
 assert read(D/'incoming_process_exit_v2_304.json')['returncode']==0
 assert read(D/'incoming_diagnostic_v2_303/completion_before_close.json')['completed']
 names=['audit_low_seat_incoming_v2_305.py','diagnose_low_seat_incoming_v2_306.py','audit_low_seat_incoming_diagnosis_306.py','render_low_seat_all_attempts_307.py','verify_low_seat_all_attempts_308.py']
 sources={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names+[Path(__file__).name]}
 state=dict(phase='running',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=sources,physics_launched=False,repair_complete=False);S.write_text(json.dumps(state,indent=2))
 with ThreadPoolExecutor(2) as pool:
  af=pool.submit(pipeline,names[:3],True);vf=pool.submit(pipeline,names[3:])
  state['audit']=af.result();S.write_text(json.dumps(state,indent=2));state['video']=vf.result()
 assert all(sha(p)==h for p,h in sources.items())
 state.update(phase='complete' if state['audit']['passed'] and state['video']['passed'] else 'needs_recovery',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources_unchanged=True,physical_admission=False)
 S.write_text(json.dumps(state,indent=2));print(json.dumps(state),flush=True)
if __name__=='__main__':main()
