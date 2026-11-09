"""Continue only unrun furniture trials, using a bounded cooperative4090 lock."""
import os, sys, json, time, fcntl, subprocess, signal, hashlib, shutil, traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent
SRC=ROOT/'experiments/furniture_examples_20261105_v2';RUN=ROOT/'outputs'/SRC.name
OUT=ROOT/'outputs'/HERE.name;PY=ROOT/'IsaacLab/.venv/bin/python'
GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def workers():return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def main():
    OUT.mkdir(exist_ok=False)
    owner=(ROOT/'outputs/furniture_scientific_owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan=read(HERE/'continuation.json');freeze=read(HERE/'freeze.json')['files'];child=None;start=time.monotonic()
    results=read(RUN/'results.json')['results'];done={r['id'] for r in results}
    assert len(done)==len(results)==4 and sorted(done)==sorted(plan['completed_ids'])
    assert read(HERE/'source_review.json')['passed']
    def verify():
        for p,h in freeze.items():assert sha(p)==h,('FROZEN_INPUT_CHANGED',p)
    def progress(state,**kw):
        write(OUT/'status.json',dict(state=state,owner_pid=os.getpid(),completed_new=sum(not r.get('reused_verified_example') for r in results),new_passes=sum(r['admitted'] and not r.get('reused_verified_example') for r in results),wall_s=time.monotonic()-start,**kw))
    try:
        verify()
        assert not read(RUN/'results.json')['complete']
        for ident in plan['remaining_ids']:assert not (RUN/ident).exists(),('UNRUN_DIRECTORY_EXISTS',ident)
        # Same read-only advisory lock used by the neighboring study. Never signal it.
        with open('/home/kadir/sircus-isaac/scene.lock','r') as shared:
            while True:
                acquired=False
                try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True;busy=workers()
                except BlockingIOError:busy='shared-lock held'
                if acquired and not busy:break
                if acquired:fcntl.flock(shared,fcntl.LOCK_UN)
                progress('WAITING_FOR_4090',reason=busy)
                assert time.monotonic()-start<1800,'BOUNDED_GPU_WAIT_EXPIRED'
                time.sleep(5)
            write(OUT/'launch_receipt.json',dict(gpu=GPU,shared_lock_held=True,freeze_sha256=sha(HERE/'freeze.json'),wait_s=time.monotonic()-start,remaining_new_trials=16,previous_complete_new_trials=4))
            for variant in read(SRC/'protocol.json')['variants']:
                ident=variant['id']
                if ident in done:continue
                if variant.get('reuse'):
                    source=Path(variant['reuse']);r=read(source/'independent_audit.json')
                    assert r['admitted'] and r['passed_integrity']
                    assert sha(source/'trajectory.npz')==r['source_hashes']['trajectory.npz']
                    results.append(dict(id=ident,admitted=True,reused_verified_example=True,source=str(source),source_audit=r));continue
                assert ident in plan['remaining_ids'];verify()
                assert time.monotonic()-start+plan['prior_owner_wall_s']<14000
                assert shutil.disk_usage(RUN).free>20_000_000_000
                assert not workers(),('FOREIGN_4090_WORKER',workers())
                folder=RUN/ident;assert not folder.exists()
                script='run_sitting.py' if variant['task']=='sitting' else 'run_leaning.py'
                cmd=['bash',str(ROOT/'scripts/run_supported_gpu0.sh'),str(SRC/script),'--variant',ident]
                if variant['task']=='leaning':cmd+=['--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
                assert not (RUN/(ident+'.command.json')).exists();write(RUN/(ident+'.command.json'),cmd);tick=time.monotonic()
                with (RUN/(ident+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    progress('RUNNING',active=ident,child_pid=child.pid)
                    while child.poll() is None:
                        time.sleep(5)
                        temp=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
                        assert temp<85,('GPU_TEMPERATURE',temp)
                        assert time.monotonic()-tick<600,('TRIAL_WALL_BUDGET',ident)
                    write(RUN/(ident+'.exit.json'),dict(returncode=child.returncode,wall_s=time.monotonic()-tick))
                if child.returncode and not (folder/'planning_failure.json').exists():raise RuntimeError(('EXECUTION_FAILURE_NO_RETRY',ident,child.returncode))
                audit=subprocess.run([str(PY),str(SRC/'audit_rollout.py'),str(folder)],cwd=ROOT,capture_output=True,text=True,timeout=90,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'))
                (RUN/(ident+'.audit.log')).write_text(audit.stdout+audit.stderr)
                assert audit.returncode==0,('AUDIT_FAILURE_NO_RETRY',ident,audit.returncode)
                result=read(folder/'independent_audit.json');results.append(dict(id=ident,**result))
                write(OUT/'results.json',dict(complete=False,results=results));progress('BETWEEN_TRIALS')
            assert len(results)==22 and sum(not r.get('reused_verified_example') for r in results)==20
            write(OUT/'results.json',dict(complete=True,results=results));progress('COMPLETE',ppo_updates=0)
    except BaseException as e:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(OUT/'results.json',dict(complete=False,results=results));progress('STOPPED',error=repr(e))
        write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
