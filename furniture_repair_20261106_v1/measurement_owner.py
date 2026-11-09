"""One finite4090 measurement job, waiting cooperatively behind the other study."""
import os,sys,json,time,fcntl,hashlib,subprocess,signal,shutil,traceback
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/HERE.name/'measurement_owner';PY=ROOT/'IsaacLab/.venv/bin/python'
GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
read=lambda p:json.loads(Path(p).read_text())
def write(p,d):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def workers():return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();child=None
    owner=(ROOT/'outputs/furniture_scientific_owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=read(HERE/'measurement_freeze.json');assert frozen['source_review']['passed']
    def progress(state,**kw):write(OUT/'status.json',dict(state=state,owner_pid=os.getpid(),wall_s=time.monotonic()-start,**kw))
    def verify():
        for p,h in frozen['files'].items():assert sha(p)==h,('FROZEN_SOURCE_CHANGED',p)
    try:
        verify();progress('WAITING_FOR_4090',scientific_launches=0)
        with open('/home/kadir/sircus-isaac/scene.lock','r') as shared:
            while True:
                acquired=False
                try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True;busy=workers()
                except BlockingIOError:busy='cooperative lock held'
                if acquired and not busy:break
                if acquired:fcntl.flock(shared,fcntl.LOCK_UN)
                progress('WAITING_FOR_4090',reason=busy,scientific_launches=0)
                if time.monotonic()-start>=10800:raise TimeoutError('FINITE_GPU_WAIT_EXPIRED')
                time.sleep(15)
            verify();assert not workers();assert shutil.disk_usage(ROOT/'outputs').free>20_000_000_000
            temp=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip());assert temp<85
            cmd=['bash',str(ROOT/'scripts/run_supported_gpu0.sh'),str(HERE/'record_motor_axis.py'),
                 '--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
            write(OUT/'command.json',cmd);tick=time.monotonic()
            with (OUT/'record.log').open('x') as log:
                child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                progress('RECORDING_MOTOR_AXIS',child_pid=child.pid,scientific_launches=1)
                while child.poll() is None:
                    time.sleep(5)
                    temp=int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
                    if temp>=85 or time.monotonic()-tick>900:raise TimeoutError('OWN_CHILD_TEMPERATURE_OR_TIME_CAP')
                write(OUT/'record_exit.json',dict(returncode=child.returncode,seconds=time.monotonic()-tick))
            assert child.returncode==0,('RECORDING_FAILED_NO_RETRY',child.returncode)
            # Release the GPU immediately; all following work is CPU-only.
            fcntl.flock(shared,fcntl.LOCK_UN)
        rollout=ROOT/'outputs'/HERE.name/'motor_axis_readout'
        audits=[('physical',ROOT/'experiments/armchair_rest_ppo_20261105_v1/audit_demo_v2.py',[str(rollout)],90),
                ('motor_axis',HERE/'audit_motor_axis.py',[],45)]
        for label,script,args,cap in audits:
            progress('AUDITING',audit=label,scientific_launches=1)
            r=subprocess.run([str(PY),str(script),*args],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=cap)
            (OUT/(label+'.log')).write_text(r.stdout+r.stderr)
            if r.returncode:raise RuntimeError(('AUDIT_REJECTED_NO_AUTOMATIC_RETRY',label,r.returncode))
        progress('MEASUREMENT_COMPLETE',scientific_launches=1,physical_admitted=read(rollout/'independent_audit.json')['admitted'],
                 motor_axis_validated=True,ppo_updates=0,new_furniture_demonstrations=0,
                 next='Separate frozen controller/trial admission required; this owner does not start PPO.')
    except BaseException as e:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        progress('STOPPED',error=repr(e));write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
