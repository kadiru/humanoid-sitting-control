"""One finite4090 owner after existing PPO/furniture jobs release their locks."""
import fcntl
import sys
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from common import ROOT,DATA,read,write,sha
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/motion_prior_ppo_20261029_v2'
PREV=ROOT/'outputs/motion_prior_ppo_20261029_v1';SOURCE=ROOT/'experiments/motion_prior_ppo_20261029_v1'
sys.path.append(str(SOURCE))
OLD=ROOT/'outputs/recovery_imitation_sitting_20261028_v3';FURN=ROOT/'outputs/furniture_validation_20261028_v2'
ASSETS=ROOT/'experiments/furniture_test_20261028_v1';BANK=ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting'
GPU='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4';PY=ROOT/'IsaacLab/.venv/bin/python'

def alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False

def main():
    protocol=read(HERE/'protocol.json');limits=protocol['resources'];bindings=read(HERE/'freeze.json')
    OUT.mkdir(exist_ok=True);own=(OUT/'owner.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'receipt.json').exists(),'NO_AUTOMATIC_RETRY'
    start=time.monotonic();execution=None;active=None;completed=[];records=[];locks=[];launches=0;state='waiting_for_existing_jobs'
    def publish(**extra):
        write(OUT/'status.json',dict(state=state,owner_pid=os.getpid(),active=active,completed=completed,
            elapsed_s=time.monotonic()-start,simulation_launches=launches,gpu='4090only',gpu4070='AIWorker untouched',recurring_automation=False,**extra))
    def frozen():
        for p,h in bindings.items():assert sha(p)==h,('FROZEN_SOURCE_OR_INPUT_CHANGED',p)
    def resource():
        assert time.monotonic()-start<limits['wait_seconds']+limits['execution_seconds']
        if execution is not None:assert time.monotonic()-execution<limits['execution_seconds']
        assert shutil.disk_usage(OUT).free>limits['free_disk_bytes']
        assert sum(p.stat().st_size for root in [OUT,PREV] for p in root.rglob('*') if p.is_file())<limits['output_bytes']
        mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        assert mem>limits['free_ram_bytes']
        if active:
            ps=subprocess.check_output(['ps','-eo','pgid=,rss='],text=True)
            assert sum(int(l.split()[1])*1024 for l in ps.splitlines() if int(l.split()[0])==active['pid'])<limits['owned_rss_bytes']
    def apps():return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    def job(label,args,gpu=False,training=False):
        nonlocal active
        resource();frozen()
        for _ in range(12):
            if not apps():break
            time.sleep(5)
        assert not apps(),'4090_NOT_IDLE'
        args=list(map(str,args));write(OUT/(label+'.command.json'),args)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        t=time.monotonic();verified=not gpu;weights=False
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(args,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active=dict(label=label,pid=child.pid);publish()
            try:
                while child.poll() is None:
                    time.sleep(5);resource();assert time.monotonic()-t<limits['child_seconds'],('CHILD_TIMEOUT',label)
                    if gpu:
                        groups={int(l.split()[0]):int(l.split()[1]) for l in subprocess.check_output(['ps','-eo','pid=,pgid='],text=True).splitlines()}
                        query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                        devices={l.split(',')[0].strip() for l in query.splitlines() if groups.get(int(l.split(',')[1]))==child.pid}
                        assert not devices or devices=={GPU},('WRONG_GPU',devices)
                        verified|=bool(devices)
                        temp,used=map(int,subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu,memory.used','--format=csv,noheader,nounits'],text=True).split(','))
                        assert temp<limits['gpu_temperature_C'] and used<limits['gpu_memory_MiB']
                    if training and not weights and (OUT/label/'weights_ready.json').exists():
                        from audit import weights as audit_weights
                        result=audit_weights(OUT/label);write(OUT/(label+'_weights_verified.json'),result);weights=True
                    publish()
                assert child.returncode==0,(label,child.returncode)
                assert verified,'GPU_NOT_VERIFIED'
                if training:assert weights,'NO_LIVE_ACTOR_AND_CRITIC_UPDATE'
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                write(OUT/(label+'.exit.json'),dict(returncode=child.returncode,seconds=time.monotonic()-t,gpu_verified=verified,weight_updates_verified=weights));active=None
        completed.append(label);publish()
    def simulation(label,script,args,training=False):
        nonlocal launches
        launches+=1;assert launches<=limits['maximum_simulator_launches']
        job(label,['bash',ROOT/'scripts/run_supported_gpu0.sh',(SOURCE/script if script=='run_training.py' else HERE/script),*args,'--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false'],True,training)
    def evaluate(label,checkpoint,seed):
        simulation(label,'run_training.py',['--mode','evaluate','--bank',BANK,'--output',OUT/label,'--checkpoint',checkpoint,'--seed',seed])
        job(label+'_audit',[PY,SOURCE/'audit.py','evaluate',OUT/label])
        a=read(OUT/label/'independent_audit.json');rows=read(OUT/label/'results.json')['episodes'];assert a['passed'] and len(rows)==72
        clean={r['id'] for r in rows if r['clean']};protected=read(SOURCE/'protected_sitting_cases.json')
        result=dict(label=label,seed=seed,clean=len(clean),repairs=sorted(clean&set(protected['failed'])),regressions=sorted(set(protected['protected'])-clean),complete_trials=72,duration_s=20)
        records.append(result);write(OUT/'comparison.json',dict(results=records,complete=False,automatic_promotion=False));return result
    try:
        frozen();previous=read(PREV/'status.json')
        assert previous['state']=='stopped' and previous['active'] is None and not alive(previous['owner_pid'])
        assert read(PREV/'prior_full72/independent_audit.json')['passed']
        baseline=read(PREV/'imitation_restoration.json');assert baseline['clean']>0
        assert read(PREV/'reward_validation.json')['passed'] and read(PREV/'construction_audit.json')['passed']
        assert read(HERE/'source_review.json')['passed']
        write(OUT/'receipt.json',dict(owner_pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=sha(HERE/'protocol.json'),freeze_sha256=sha(HERE/'freeze.json'),recurring_automation=False))
        for p in [DATA/'contact_parallel_gpu_v1/transition.lock',DATA/'contact_ppo_scientific_owner.lock']:
            f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        execution=time.monotonic();assert not apps(),'4090_BUSY'
        write(OUT/'dependency_results.json',dict(previous=previous,baseline=baseline,completed_steps_not_repeated=True))
        records.append(baseline)
        state='body_part_support_tests';publish()
        simulation('body_support','body_support_probe.py',['--output',OUT/'body_support','--inventory',HERE/'body_colliders.json'])
        job('body_support_audit',[PY,HERE/'audit_support.py','--output',OUT/'body_support'])
        state='ppo';publish()
        for seed in [64111,64117]:
            label=f'train_sitting_{seed}'
            simulation(label,'run_training.py',['--mode','train','--bank',BANK,'--output',OUT/label,'--seed',seed,
                '--checkpoint',PREV/f'prepared/initial_{seed}.pt','--demonstrations',PREV/'prepared/demonstrations.npz','--updates',256],True)
            for update in [128,256]:evaluate(f'eval_{seed}_{update}',OUT/label/f'update_{update:03d}.pt',seed)
        state='complete';write(OUT/'comparison.json',dict(results=records,complete=True,automatic_promotion=False));publish()
    except BaseException as e:
        state='stopped';write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=completed));publish(error=repr(e));raise

if __name__=='__main__':main()
