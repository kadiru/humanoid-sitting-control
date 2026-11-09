"""One bounded camera-validation job; not a recurring automation or motion run."""
import os,sys,json,time,hashlib,fcntl,signal,subprocess,shutil,datetime
from pathlib import Path
R=Path('/home/kadir/humanoid-sitting');H=Path(__file__).resolve().parent;O=R/'outputs'/H.name
PY=R/'IsaacLab/.venv/bin/python';GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
def write(p,d):
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(p)
def gpu(query):return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-'+query,'--format=csv,noheader,nounits'],text=True).strip()
def main():
    start=time.monotonic();child=None;status=dict(owner_pid=os.getpid(),scientific_launches=0,simulation_launches=0,PPO_updates=0,recurring=False,queue_deadline_s=7200)
    def report(state,**kw):
        status.update(state=state,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),wall_s=time.monotonic()-start,**kw);write(O/'queued_capture_status.json',status)
    def run(cmd,log_path,seconds):
        nonlocal child
        tick=time.monotonic()
        with log_path.open('x') as log:
            child=subprocess.Popen(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            while child.poll() is None:
                time.sleep(2)
                if time.monotonic()-tick>seconds:raise TimeoutError('OWN_CHILD_DEADLINE')
                if seconds==900 and int(gpu('gpu=temperature.gpu'))>=85:raise TimeoutError('OWN_GPU_TEMPERATURE_LIMIT')
        code=child.returncode;child=None
        if code:raise RuntimeError(('CHILD_FAILED_NO_RETRY',code,str(log_path)))
        return time.monotonic()-tick
    try:
        O.mkdir(exist_ok=True)
        queue_lock=(R/'outputs/narrow_fixture_capture_queue.lock').open('a');fcntl.flock(queue_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert not (O/'camera_and_wrench_capture').exists(),'NO_CAPTURE_REPLAY'
        freeze=json.loads((H/'queue_freeze.json').read_text())
        for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        assert json.loads((H/'preflight.json').read_text())['passed']
        assert json.loads((R/'experiments/rest_cycle_quiet_arms_20261109_v1/nominal_construction/result.json').read_text())['nominal_model_passed']
        science=(R/'outputs/furniture_scientific_owner.lock').open('a');shared=Path('/home/kadir/sircus-isaac/scene.lock').open('r')
        report('WAITING_FOR_4090_AND_SHARED_LOCK',purpose='One standing camera capture and independent depth/region audits only; no sitting, PPO, forced job interruption or recurring automation.')
        while True:
            if time.monotonic()-start>=7200:raise TimeoutError('FINITE_QUEUE_EXPIRED_NO_LAUNCH')
            try:
                fcntl.flock(science,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB)
                if gpu('compute-apps=pid'):raise BlockingIOError('4090_BUSY')
                break
            except BlockingIOError:
                fcntl.flock(shared,fcntl.LOCK_UN);fcntl.flock(science,fcntl.LOCK_UN);time.sleep(5)
        for p,h in freeze['files'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        assert int(gpu('gpu=temperature.gpu'))<85 and shutil.disk_usage(R/'outputs').free>20_000_000_000
        cmd=['bash',str(R/'scripts/run_supported_gpu0.sh'),str(H/'capture_view.py'),'--name','sit_modern_coffee_table_01_center_normal_landing','--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
        write(O/'queued_capture_command.json',cmd);report('CAPTURING_STANDING_CAMERA',scientific_launches=1,simulation_launches=1)
        seconds=run(cmd,O/'queued_trial.log',900);fcntl.flock(shared,fcntl.LOCK_UN)
        s=O/'camera_and_wrench_capture';inputs=dict(freeze['files'])
        inputs.update({str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in s.iterdir() if p.is_file()})
        write(H/'audit_freeze.json',dict(files=inputs,source_freeze_sha256=hashlib.sha256((H/'queue_freeze.json').read_bytes()).hexdigest()))
        report('AUDITING_CURRENT_DEPTH',capture_seconds=seconds,scientific_launches=2)
        dep_seconds=run([str(PY),str(H/'audit_capture.py')],O/'queued_depth_audit.log',46)
        assert json.loads((s/'capture_audit.json').read_text())['passed'],'CURRENT_DEPTH_AUDIT_FAILED'
        inputs[str(s/'capture_audit.json')]=hashlib.sha256((s/'capture_audit.json').read_bytes()).hexdigest()
        write(H/'patch_v2_freeze.json',dict(files=inputs,source_freeze_sha256=hashlib.sha256((H/'queue_freeze.json').read_bytes()).hexdigest()))
        report('AUDITING_CURRENT_SUPPORT_POLYGON',depth_audit_seconds=dep_seconds,scientific_launches=3)
        region_seconds=run([str(PY),str(H/'audit_patch.py')],O/'queued_region_audit.log',46)
        assert json.loads((s/'patch_audit_v2.json').read_text())['passed'],'CURRENT_REGION_AUDIT_FAILED'
        report('CAMERA_AND_REGION_VALIDATED_REQUIRES_MOTION_LOAD_VALIDATION',region_audit_seconds=region_seconds,sitting_controls=0,complete_cycle=False)
    except BaseException as e:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=10)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        report('STOPPED_NO_RETRY',error=repr(e));raise
if __name__=='__main__':main()
