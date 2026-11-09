"""Finite sequential4090 collection; never starts PPO from unqualified demos."""
import os,sys,json,time,fcntl,hashlib,subprocess,signal,shutil,traceback
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/HERE.name/'calibration_owner';PY=ROOT/'IsaacLab/.venv/bin/python'
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
def temperature():return int(subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu','--format=csv,noheader,nounits'],text=True).strip())
def main():
    OUT.mkdir(parents=True,exist_ok=False);start=time.monotonic();child=None;rows=[]
    owner=(ROOT/'outputs/furniture_scientific_owner.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=read(HERE/'calibration_freeze.json');assert frozen['source_review']['passed']
    jobs=read(HERE/'calibration_jobs.json')
    def progress(state,**kw):write(OUT/'status.json',dict(state=state,owner_pid=os.getpid(),wall_s=time.monotonic()-start,completed=rows,**kw))
    def verify():
        for p,h in frozen['files'].items():assert sha(p)==h,('FROZEN_SOURCE_CHANGED',p)
    try:
        verify();progress('WAITING_FOR_4090',scientific_launches=0)
        with open('/home/kadir/sircus-isaac/scene.lock','r') as shared:
            for job in jobs:
                while True:
                    acquired=False
                    try:fcntl.flock(shared,fcntl.LOCK_EX|fcntl.LOCK_NB);acquired=True;busy=workers()
                    except BlockingIOError:busy='cooperative lock held'
                    if acquired and not busy:break
                    if acquired:fcntl.flock(shared,fcntl.LOCK_UN)
                    progress('WAITING_FOR_4090',reason=busy,next_case=job['name'],scientific_launches=len(rows))
                    if time.monotonic()-start>=57600:raise TimeoutError('FINITE16H_TOTAL_OWNER_CAP')
                    time.sleep(15)
                verify();assert not workers();assert shutil.disk_usage(ROOT/'outputs').free>20_000_000_000
                assert temperature()<85
                cmd=['bash',str(ROOT/'scripts/run_supported_gpu0.sh'),str(HERE/'calibration_demo.py'),
                     '--name',job['name'],'--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
                write(OUT/(job['name']+'.command.json'),cmd);tick=time.monotonic()
                with (OUT/(job['name']+'.log')).open('x') as log:
                    child=subprocess.Popen(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    progress('COLLECTING',case=job['name'],child_pid=child.pid,scientific_launches=len(rows)+1)
                    while child.poll() is None:
                        time.sleep(5)
                        if temperature()>=85 or time.monotonic()-tick>900 or time.monotonic()-start>57600:
                            raise TimeoutError('OWN_CHILD_TEMPERATURE_OR_TIME_CAP')
                row=dict(case=job['name'],returncode=child.returncode,seconds=time.monotonic()-tick)
                fcntl.flock(shared,fcntl.LOCK_UN)
                rollout=ROOT/'outputs'/HERE.name/'calibration_demos'/job['name']
                if child.returncode==0:
                    progress('AUDITING',case=job['name'],scientific_launches=len(rows)+1)
                    r=subprocess.run([str(PY),str(HERE/'audit_rollout.py'),str(rollout)],cwd=ROOT,
                        env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='1'),capture_output=True,text=True,timeout=90)
                    (OUT/(job['name']+'.audit.log')).write_text(r.stdout+r.stderr)
                    row['audit_returncode']=r.returncode
                    if r.returncode==0:row['audit']=read(rollout/'independent_audit.json')
                rows.append(row);write(OUT/(job['name']+'.exit.json'),row)
                # No retry. Other independently frozen starts continue once each.
            progress('COLLECTION_COMPLETE',scientific_launches=len(rows),
                admitted=sum(bool(r.get('audit',{}).get('admitted')) for r in rows),
                ppo_updates=0,next='Audit load coverage and complete the20-demo corpus before PPO admission.')
    except BaseException as e:
        if child and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=20)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
        progress('STOPPED',error=repr(e));write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc()));raise
if __name__=='__main__':main()
