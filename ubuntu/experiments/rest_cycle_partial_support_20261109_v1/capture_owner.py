"""One bounded4090 sitting collection, followed by independent CPU audits."""
import os,json,time,fcntl,hashlib,subprocess,signal,shutil,sys,datetime
from pathlib import Path
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent
OUT=R/'outputs'/H.name;PY=R/'IsaacLab/.venv/bin/python';GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def gpu(field):
    return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-'+field,'--format=csv,noheader,nounits'],text=True).strip()
def main():
    OUT.mkdir(exist_ok=False);start=time.monotonic();child=None;launched=False
    status={'owner_pid':os.getpid(),'scientific_launches':0,'PPO_updates':0,'retries':0,'GPU':GPU}
    def report(state,**kw):
        status.update(state=state,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_seconds=time.monotonic()-start,**kw);write(OUT/'owner_status.json',status)
    try:
        lock=(R/'outputs/furniture_scientific_owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        shared=Path('/home/kadir/sircus-isaac/scene.lock').open('r');fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
        frozen=json.loads((H/'capture_freeze.json').read_text());assert frozen['source_review_passed']
        for name,h in frozen['files'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h, ('FROZEN_FILE_CHANGED',name)
        assert json.loads((H/'camera_preflight.json').read_text())['passed']
        assert not gpu('compute-apps=pid'), '4090_BUSY'
        assert int(gpu('gpu=temperature.gpu'))<85
        assert shutil.disk_usage(R/'outputs').free>20_000_000_000
        cmd=['bash',str(R/'scripts/run_supported_gpu0.sh'),str(H/'capture_view.py'),'--name','sit_modern_coffee_table_01_center_normal_landing',
             '--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
        write(OUT/'command.json',cmd);tick=time.monotonic()
        with (OUT/'trial.log').open('x') as log:
            child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);launched=True
            report('CAPTURING_CAMERA_AND_SHADOW_JOINT_WRENCH',child_pid=child.pid,scientific_launches=1)
            while child.poll() is None:
                time.sleep(3)
                if time.monotonic()-tick>900 or int(gpu('gpu=temperature.gpu'))>=85:raise TimeoutError('OWN_CHILD_RESOURCE_CAP')
        report('TRIAL_EXITED',trial_returncode=child.returncode,trial_seconds=time.monotonic()-tick)
        fcntl.flock(shared,fcntl.LOCK_UN)
        assert child.returncode==0, 'TRIAL_FAILED_NO_RETRY'
        report('CAPTURE_COMPLETE_REQUIRES_INDEPENDENT_AUDIT',sitting_controls=0,complete_cycle=False)
    except BaseException as e:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        report('STOPPED_NO_RETRY',error=repr(e),scientific_launches=int(launched));raise
if __name__=='__main__':main()
