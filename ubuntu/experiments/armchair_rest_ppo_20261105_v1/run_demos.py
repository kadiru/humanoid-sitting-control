"""Finite fourteen-trial owner; scientific failures preserved, no automatic retry."""
import os,sys,fcntl,time,subprocess,signal,shutil,traceback
from pathlib import Path
from common import ROOT,DATA,read,write,sha
from audit_demo import check
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/armchair_rest_ppo_20261105_v1';GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
def main():
    OUT.mkdir(exist_ok=True);assert not (OUT/'demo_receipt.json').exists(),'NO_AUTOMATIC_RETRY'
    locks=[]
    for p in [DATA/'contact_parallel_gpu_v1/transition.lock',DATA/'contact_ppo_scientific_owner.lock']:
        f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
    freeze=read(HERE/'demo_freeze.json');protocol=read(HERE/'protocol.json');start=time.monotonic();active=None;completed=[];results=[]
    def verify():
        for p,h in freeze.items():assert sha(p)==h,('SOURCE_OR_INPUT_CHANGED',p)
    def status(state,**kw):write(OUT/'demo_status.json',dict(state=state,owner_pid=os.getpid(),active=active,completed=completed,elapsed_s=time.monotonic()-start,**kw))
    verify();assert read(OUT/'preflight.json')['passed']
    write(OUT/'demo_receipt.json',dict(owner_pid=os.getpid(),source_freeze_sha256=sha(HERE/'demo_freeze.json'),protocol_sha256=sha(HERE/'protocol.json'),scientific_owner_count=1))
    try:
        for profile in protocol['demonstrations']['profiles']:
            for i,var in enumerate(protocol['demonstrations']['variants']):
                verify();assert time.monotonic()-start<10000
                assert shutil.disk_usage(OUT).free>20_000_000_000
                assert not subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader']).strip(),'4090_BUSY'
                name=f"demo_{profile}_{var['id']}";out=OUT/name
                cmd=['bash',str(ROOT/'scripts/run_supported_gpu0.sh'),str(HERE/'demo.py'),'--output',str(out),'--variant',str(i),'--profile',profile,'--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
                write(OUT/(name+'.command.json'),cmd);t=time.monotonic()
                with (OUT/(name+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'))
                    active=dict(pid=child.pid,label=name);status('demonstrations')
                    while child.poll() is None:
                        time.sleep(5)
                        temp=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits']).strip());assert temp<85,('GPU_TEMPERATURE',temp)
                        if time.monotonic()-t>600:os.killpg(child.pid,signal.SIGTERM);child.wait(timeout=25);raise TimeoutError(name)
                    write(OUT/(name+'.exit.json'),dict(returncode=child.returncode,seconds=time.monotonic()-t))
                active=None
                if child.returncode and not (out/'planning_failure.json').exists():raise RuntimeError((name,child.returncode))
                result=check(out);results.append(dict(label=name,**result));completed.append(name)
                write(OUT/'demo_results.json',dict(complete=False,results=results));status('demonstrations')
        write(OUT/'demo_results.json',dict(complete=True,results=results));status('complete',admitted=sum(x['admitted'] for x in results))
    except BaseException as e:
        if active and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        status('stopped',error=repr(e));write(OUT/'demo_failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
