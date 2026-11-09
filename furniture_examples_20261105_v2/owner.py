"""Finite, sequential4090-only collection. No background retry or scheduler."""
import os,sys,json,time,fcntl,subprocess,signal,hashlib,shutil,traceback
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent.parent;OUT=ROOT/'outputs'/HERE.name
GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4';PY=ROOT/'IsaacLab/.venv/bin/python'
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):
    p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(d,indent=2,allow_nan=False)+'\n');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def gpu_workers():return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def main():
    assert read(HERE/'static_audit.json')['passed'] and read(HERE/'source_review.json')['passed']
    OUT.mkdir(exist_ok=False);lock=(ROOT/'outputs/furniture_scientific_owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    freeze=read(HERE/'freeze.json')['files'];protocol=read(HERE/'protocol.json');start=time.monotonic();child=None;results=[]
    def verify():
        for p,h in freeze.items():assert sha(p)==h,('FROZEN_INPUT_CHANGED',p)
    try:
        verify();assert not gpu_workers(),('4090_BUSY',gpu_workers())
        write(OUT/'receipt.json',dict(owner_pid=os.getpid(),protocol_sha256=sha(HERE/'protocol.json'),freeze_sha256=sha(HERE/'freeze.json'),gpu=GPU,recurring_automation=False))
        for variant in protocol['variants']:
            if variant.get('reuse'):
                source=Path(variant['reuse']);r=read(source/'independent_audit.json')
                assert r['admitted'] and r['passed_integrity']
                assert sha(source/'trajectory.npz')==r['source_hashes']['trajectory.npz']
                results.append(dict(id=variant['id'],admitted=True,reused_verified_example=True,source=str(source),source_audit=r))
                write(OUT/'results.json',dict(complete=False,results=results))
                continue
            verify();assert time.monotonic()-start<14000;assert shutil.disk_usage(OUT).free>20_000_000_000
            assert not gpu_workers(),('4090_BUSY',gpu_workers())
            ident=variant['id'];folder=OUT/ident
            script='run_sitting.py' if variant['task']=='sitting' else 'run_leaning.py'
            cmd=['bash',str(ROOT/'scripts/run_supported_gpu0.sh'),str(HERE/script),'--variant',ident]
            if variant['task']=='leaning':cmd+=['--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
            write(OUT/(ident+'.command.json'),cmd);tick=time.monotonic()
            with (OUT/(ident+'.log')).open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                write(OUT/'status.json',dict(state='RUNNING',owner_pid=os.getpid(),child_pid=child.pid,active=ident,completed=len(results),admitted=sum(r['admitted'] for r in results)))
                while child.poll() is None:
                    time.sleep(5)
                    temp=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
                    assert temp<85,('GPU_TEMPERATURE',temp)
                    assert time.monotonic()-tick<600,('TRIAL_WALL_BUDGET',ident)
                write(OUT/(ident+'.exit.json'),dict(returncode=child.returncode,wall_s=time.monotonic()-tick))
            if child.returncode and not (folder/'planning_failure.json').exists():raise RuntimeError(('EXECUTION_FAILURE_NO_RETRY',ident,child.returncode))
            audit=subprocess.run([str(PY),str(HERE/'audit_rollout.py'),str(folder)],cwd=ROOT,capture_output=True,text=True,timeout=90,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'))
            (OUT/(ident+'.audit.log')).write_text(audit.stdout+audit.stderr)
            assert audit.returncode==0,('AUDIT_FAILURE_NO_RETRY',ident,audit.returncode)
            result=read(folder/'independent_audit.json');results.append(dict(id=ident,**result));write(OUT/'results.json',dict(complete=False,results=results))
        write(OUT/'results.json',dict(complete=True,results=results,positive_sitting=sum(r['admitted'] and r['id'].startswith('sit_') for r in results),positive_leaning=sum(r['admitted'] and r['id'].startswith('lean_') for r in results)))
        write(OUT/'status.json',dict(state='COMPLETE',completed=len(results),admitted=sum(r['admitted'] for r in results),wall_s=time.monotonic()-start,ppo_updates=0))
    except BaseException as e:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        write(OUT/'status.json',dict(state='STOPPED',error=repr(e),completed=len(results),admitted=sum(r['admitted'] for r in results),wall_s=time.monotonic()-start))
        write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
