"""One finite dependency job, never a recurring automation or PPO interrupter."""
import fcntl
import os
import signal
import shutil
import subprocess
import time
import traceback
from io_utils import ROOT,HERE,OUT,ASSETS,TRAIN,PPO,GPU,PY,read,write,sha

def main():
    OUT.mkdir(exist_ok=True)
    lock=(OUT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'receipt.json').exists(),'NO_AUTOMATIC_RETRY'
    protocol=read(HERE/'protocol.json');bindings=read(HERE/'freeze.json')
    started=time.monotonic();execution_start=None;active=None;completed=[];locks=[];state='waiting_for_PPO'
    def publish(**extra):
        write(OUT/'status.json',dict(state=state,owner_pid=os.getpid(),active=active,completed=completed,
            waiting_seconds=time.monotonic()-started,gpu='4090 only',gpu4070='AIWorker untouched',
            recurring_automation=False,**extra))
    def freeze():
        for p,s in bindings.items():assert sha(p)==s,('FROZEN_INPUT_CHANGED',p)
    def resources():
        assert time.monotonic()-started<protocol['wait_cap_seconds']+protocol['execution_cap_seconds']
        if execution_start is not None:assert time.monotonic()-execution_start<protocol['execution_cap_seconds']
        assert shutil.disk_usage(OUT).free>8*1024**3
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<8*1024**3
        mem=next(int(l.split()[1])*1024 for l in __import__('pathlib').Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        assert mem>8*1024**3
        if active:
            ps=subprocess.check_output(['ps','-eo','pgid=,rss='],text=True)
            assert sum(int(l.split()[1])*1024 for l in ps.splitlines() if int(l.split()[0])==active['pid'])<24*1024**3
    def apps():
        return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    def job(label,args,gpu):
        nonlocal active
        freeze();resources()
        for idle_wait in range(12):
            if not apps():break
            time.sleep(5)
        assert not apps(), '4090_NOT_IDLE'
        args=list(map(str,args));write(OUT/(label+'.command.json'),args)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        t=time.monotonic();verified=not gpu
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active=dict(label=label,pid=child.pid);publish()
            try:
                while child.poll() is None:
                    time.sleep(5);resources();assert time.monotonic()-t<3600,'CHILD_WALL_LIMIT'
                    if gpu:
                        groups={int(l.split()[0]):int(l.split()[1]) for l in subprocess.check_output(['ps','-eo','pid=,pgid='],text=True).splitlines()}
                        query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                        devices={l.split(',')[0].strip() for l in query.splitlines() if groups.get(int(l.split(',')[1]))==child.pid}
                        assert not devices or devices=={GPU},('WRONG_GPU',devices)
                        verified|=bool(devices)
                        readings=subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu,memory.used','--format=csv,noheader,nounits'],text=True)
                        temp,used=map(int,readings.split(','));assert temp<85 and used<23552
                    publish()
                assert child.returncode==0,(label,child.returncode)
                assert verified,'GPU_NOT_VERIFIED'
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                write(OUT/(label+'.exit.json'),dict(returncode=child.returncode,seconds=time.monotonic()-t,gpu_verified=verified))
                active=None
        completed.append(label);publish()
    def simulation(label,script,extra):
        job(label,['bash',ROOT/'scripts/run_supported_gpu0.sh',HERE/script,*extra,
            '--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false'],True)
    try:
        freeze();assert read(HERE/'preflight.json')['passed'];assert read(HERE/'source_review.json')['passed']
        write(OUT/'receipt.json',dict(owner_pid=os.getpid(),protocol_sha256=sha(HERE/'protocol.json'),
            freeze_sha256=sha(HERE/'freeze.json'),created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            authorized_dependency='after BOTH PPO seeds and all72-case evaluations finish',recurring_automation=False))
        publish()
        while True:
            s=read(PPO/'status.json');assert s['state']!='stopped',('DEPENDENCY_FAILED',s)
            if s['state']=='complete' and not s.get('active'):
                assert read(PPO/'comparison.json')['complete']
                try:os.kill(s['owner_pid'],0);owner_live=True
                except ProcessLookupError:owner_live=False
                if not owner_live:break
            resources();assert time.monotonic()-started<protocol['wait_cap_seconds'],'WAIT_CAP'
            time.sleep(30)
        execution_start=time.monotonic()
        data=__import__('pathlib').Path('/data/humanoid-sitting-perception/low_seat_contact_01')
        for path in [data/'contact_parallel_gpu_v1/transition.lock',data/'contact_ppo_scientific_owner.lock']:
            f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        assert not apps(), '4090_BUSY_AFTER_PPO'
        for seed in [64111,64117]:
            for u in [128,256]:assert read(PPO/f'eval_sitting_{seed}_{u}/independent_audit.json')['passed']
        sitting=PPO/'train_sitting_64111/update_256.pt'
        leaning=ROOT/'outputs/full_body_contact_imitation_v5/train_leaning_64111/update_256.pt'
        selected={str(p):sha(p) for p in [sitting,leaning]}
        write(OUT/'policy_bindings.json',selected);bindings.update(selected)
        write(OUT/'execution_bindings.json',bindings)
        state='collision_probes';publish()
        simulation('probes','probe.py',['--output',OUT/'probes'])
        job('probe_audit',[PY,HERE/'audit_probes.py'],False)
        allowed=read(OUT/'probes/independent_audit.json')['validated_layouts']
        state='robot_trials';publish()
        for c in read(ASSETS/'cases.json'):
            if c['id'] in allowed:
                simulation('robot_'+c['id'],'robot_trial.py',['--case',c['id'],'--output',OUT/('robot_'+c['id'])])
        state='rendering';publish();job('video',[PY,HERE/'render_video.py'],False)
        state='complete';publish(video=str(OUT/'furniture_collisions_and_forces.mp4'),
            collision_validated_layouts=allowed,robot_trials=len(allowed),physical_repair_claimed=False)
    except BaseException as e:
        state='stopped';write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=completed))
        publish(error=repr(e));raise

if __name__=='__main__':main()
