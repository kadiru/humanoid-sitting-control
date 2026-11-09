"""Single finite 4090 calibration owner. No retries or recurring task."""
import os,sys,json,time,hashlib,fcntl,signal,subprocess,shutil,datetime
from pathlib import Path
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent;O=R/'outputs'/H.name
GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def gpu(q):return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-'+q,'--format=csv,noheader,nounits'],text=True).strip()
def main():
    started=time.monotonic();child=None
    def report(state,**kw):
        write(O/'owner_status.json',dict(state=state,pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_s=time.monotonic()-started,recurring=False,**kw))
    def run(cmd,name,budget):
        nonlocal child
        stamp=time.monotonic()
        with (O/(name+'.log')).open('x') as f:
            child=subprocess.Popen(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                time.sleep(2)
                if time.monotonic()-stamp>budget:raise TimeoutError('OWN_CHILD_BUDGET')
                if budget==900 and int(gpu('gpu=temperature.gpu'))>=85:raise TimeoutError('OWN_GPU_TEMPERATURE')
        rc=child.returncode;child=None
        write(O/(name+'_process.json'),dict(returncode=rc,wall_s=time.monotonic()-stamp))
        if rc:raise RuntimeError(('CHILD_FAILED_NO_RETRY',name,rc))
    O.mkdir(exist_ok=False)
    try:
        science=(R/'outputs/furniture_scientific_owner.lock').open('a');fcntl.flock(science,fcntl.LOCK_EX|fcntl.LOCK_NB)
        shared=Path('/home/kadir/sircus-isaac/scene.lock').open('r')
        freeze=json.loads((H/'inspection_freeze.json').read_text())
        for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        report('WAITING_FOR_FREE_4090',maximum_wait_seconds=60)
        while True:
            if time.monotonic()-started>60:raise TimeoutError('FINITE_QUEUE_EXPIRED')
            try:
                fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if gpu('compute-apps=pid'):raise BlockingIOError('4090_BUSY')
                break
            except BlockingIOError:fcntl.flock(shared,fcntl.LOCK_UN);time.sleep(5)
        assert int(gpu('gpu=temperature.gpu'))<85 and shutil.disk_usage(R).free>20_000_000_000
        cmd=['bash',str(R/'scripts/run_supported_gpu0.sh'),str(H/'inspect_runtime.py'),'--name','sit_modern_coffee_table_01_center_normal_landing','--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
        write(O/'command.json',cmd);report('INSPECTING_LIVE_PROPERTIES',GPU_launches=1,PPO_updates=0)
        run(cmd,'trial',900);fcntl.flock(shared,fcntl.LOCK_UN)
        result=json.loads((O/'runtime_inspection/runtime_properties.json').read_text())
        report('LIVE_PROPERTIES_RECORDED',GPU_launches=1,commanded_sitting_controls=0,complete_cycles=0,PPO_updates=0)
    except BaseException as e:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        report('STOPPED_NO_RETRY',error=repr(e));raise

if __name__=='__main__':main()
