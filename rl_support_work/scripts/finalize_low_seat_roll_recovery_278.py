"""Complete versioned post-run audits after original278; never rerun physics/video."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess,time,traceback
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');S=D/'roll_audit_recovery_status_278.json'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 assert not S.exists();state=dict(phase='waiting_original_finalizer',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),physics_rerun=False,video_rerun=False);S.write_text(json.dumps(state,indent=2))
 while True:
  p=D/'roll_finalization_status_278.json'
  if p.exists() and read(p)['phase'] in ['complete','needs_recovery']:break
  if (D/'status_roll_feedback_probe_273.json').exists() and read(D/'status_roll_feedback_probe_273.json')['phase']=='failed':raise RuntimeError('Physics failed; no automatic retry')
  time.sleep(5)
 original=read(p);assert original['phase']=='needs_recovery' and original['video']['passed']
 assert not original['audit']['passed'] and len(original['audit']['steps'])==1
 rec=D/'roll_audit_execution_recovery_278';rec.mkdir(exist_ok=False)
 for f in [D/'roll_finalization_status_278.json',D/'roll_audit_status_278.json',D/'audit_low_seat_roll_274.py.log']:
  (rec/f.name).write_bytes(f.read_bytes())
 names=['audit_low_seat_roll_v2_274.py','audit_low_seat_roll_dual_v2_275.py','diagnose_low_seat_roll_279.py'];sources={str(R/'scripts'/n):sha(R/'scripts'/n) for n in names+[Path(__file__).name]}
 env=dict(os.environ,PYTHONPATH='scripts:sitting_task/src',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',CUDA_VISIBLE_DEVICES='1');state['phase']='running';state['steps']=[];S.write_text(json.dumps(state,indent=2))
 for n in names:
  args=['--folder',str(D/'roll_feedback_probe_273')] if 'dual' in n else []
  with (D/(n+'.log')).open('x') as log:p=subprocess.run([str(R/'IsaacLab/.venv/bin/python'),str(R/'scripts'/n),*args],cwd=R,env=env,stdout=log,stderr=subprocess.STDOUT)
  state['steps'].append(dict(script=n,exit_code=p.returncode));S.write_text(json.dumps(state,indent=2))
  if p.returncode:state['phase']='failed';S.write_text(json.dumps(state,indent=2));raise RuntimeError(n)
 assert all(sha(p)==h for p,h in sources.items())
 assert all(sha(p)==h for p,h in read(D/'roll_feedback_probe_273/revision.json')['sources'].items())
 state.update(phase='complete',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),sources_unchanged=True,source_sha256=sources,physical_admission=False,repair_complete=False)
 S.write_text(json.dumps(state,indent=2));(rec/'recovery.json').write_text(json.dumps(state,indent=2));(rec/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in rec.iterdir()},indent=2));print(json.dumps(state),flush=True)
if __name__=='__main__':main()
