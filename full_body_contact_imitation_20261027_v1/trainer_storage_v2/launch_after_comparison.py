"""One finite dependency job, not a scheduled/recurring automation.

Waits for all four128-update comparisons AND their evaluations/audits, then
owns both simulator lanes. Failure closes this attempt; there is no retry.
"""
import fcntl
import os
import signal
import subprocess
import threading
import time
import traceback
import shutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import ROOT, DATA, read, write, sha

HERE=Path(__file__).resolve().parent
PYTHON=ROOT/'IsaacLab/.venv/bin/python'
OLD=DATA/'contact_parallel_gpu_v1'
OUT=ROOT/'outputs/full_body_contact_imitation_v2'
UUIDS=['GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4','GPU-cde23916-c5fa-f18e-a343-127f732cc2f6']


def dependency_ready(status,results):
    if status['state']=='stopped':raise RuntimeError('Prior comparison stopped; dependency not satisfied')
    if status['state']!='comparison_complete' or status.get('complete') is not True:return False
    expected={(arm,seed) for arm in ['original','revised'] for seed in [63111,63117]}
    actual=[(r['arm'],r['seed']) for r in results['results'] if r['updates']==128]
    assert set(actual)==expected and len(actual)==4
    assert not status.get('active')
    return True


def main():
    protocol=read(HERE/'protocol.json'); bindings=read(HERE/'freeze.json')
    OUT.mkdir(exist_ok=True)
    own=(OUT/'owner.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'receipt.json').exists(),'ONE_FINITE_ATTEMPT_ONLY'
    assert read(HERE/'cpu_tests.json')['passed'] and read(HERE/'source_review.json')['passed']
    started=time.time(); active={};completed=[];failures=[];mutex=threading.RLock();stop=threading.Event();deadline=None
    state='waiting_for_comparison';lockfiles=[]
    def freeze():
        for path,h in bindings.items():assert sha(path)==h,('FROZEN_INPUT_CHANGED',path)
    def publish(**extra):
        with mutex:
            write(OUT/'status.json',dict(state=state,pid=os.getpid(),active={str(g):dict(pid=p.pid,label=label) for g,(p,label) in active.items()},
              completed=completed.copy(),failures=failures.copy(),seconds=time.time()-started,
              training_started=any(x.startswith('train_') for x in completed) or any(label.startswith('train_') for _,label in active.values()),
              comparison_untouched=True,recurring_automation=False,physical_repair_claimed=False,**extra))
    def resources():
        if deadline is not None:assert time.time()<deadline,'TOTAL_EXECUTION_WALL_CAP'
        size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file() and not p.name.endswith('.tmp'))
        assert size<protocol['max_output_bytes'],'OUTPUT_CAP'
        assert shutil.disk_usage(OUT).free>protocol['min_free_bytes'],'DISK_RESERVE'
        memory=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        assert memory>protocol['min_available_memory_bytes'],'RAM_RESERVE'
        # Charge only owned process groups; never signal an unrelated worker.
        ps=subprocess.check_output(['ps','-eo','pgid=,rss='],text=True)
        with mutex:groups={p.pid for p,_ in active.values()}
        rss=sum(int(l.split()[1])*1024 for l in ps.splitlines() if int(l.split()[0]) in groups)
        assert rss<protocol['max_owned_rss_bytes'],'OWNED_RSS_CAP'
        if deadline is not None:
            gpu_status=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True)
            for line in gpu_status.splitlines():
                i,memory,temperature=map(int,line.split(','))
                assert memory<protocol['gpu_memory_cap_MiB'][i],('GPU_MEMORY_CAP',i,memory)
                assert temperature<protocol['gpu_temperature_cap_C'],('GPU_TEMPERATURE_CAP',i,temperature)
        return size
    def check_gpu(p,gpu):
        groups={int(l.split()[0]):int(l.split()[1]) for l in subprocess.check_output(['ps','-eo','pid=,pgid='],text=True).splitlines()}
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        found=[line.split(',')[0].strip() for line in apps.splitlines() if groups.get(int(line.split(',')[1]))==p.pid]
        assert found and set(found)=={UUIDS[gpu]},('WRONG_GPU',gpu,found)
    def job(gpu,label,command,cap,simulation=False,training=False):
        assert not stop.is_set();freeze();resources()
        write(OUT/(label+'.command.json'),command)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        if not simulation:env['CUDA_VISIBLE_DEVICES']=''
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            with mutex:active[gpu]=(child,label)
            t=time.monotonic();checked=False;publish()
            try:
                while child.poll() is None:
                    if stop.wait(5):raise RuntimeError('Another lane failed')
                    resources();assert time.monotonic()-t<cap,('CHILD_WALL_CAP',label)
                    if training and not checked and (OUT/label/'weights_ready.json').exists():
                        check_gpu(child,gpu)
                        from audit import weights
                        audit=weights(OUT/label);assert audit['passed'];checked=True
                        write(OUT/(label+'_verified.json'),dict(gpu=gpu,**audit))
                    publish()
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
        with mutex:active.pop(gpu,None)
        write(OUT/(label+'.exit.json'),dict(returncode=child.returncode,seconds=time.monotonic()-t,simulation=simulation,gpu=gpu))
        assert child.returncode==0,(label,child.returncode)
        if training:
            assert checked,'NO_VERIFIED_LIVE_WEIGHT_UPDATE'
            assert read(OUT/label/'status.json')['updates'] in [2,256]
        with mutex:completed.append(label)
        freeze();publish()
    def sim_command(gpu,entry,args):
        wrapper=ROOT/'scripts'/('run_supported_gpu0.sh' if gpu==0 else 'run_supported_gpu1_parallel_v1.sh')
        return ['bash',str(wrapper),str(HERE/entry),*map(str,args)]
    def trainer(gpu,label,task,mode,seed,checkpoint=None,updates=256,reset=None):
        args=['--bank',OUT/'dataset'/task,'--output',OUT/label,'--mode',mode,'--seed',seed,'--updates',updates,
              '--device','cuda:0',f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']
        if checkpoint:args+=['--checkpoint',checkpoint]
        if reset:args+=['--reset-audit',reset]
        job(gpu,label,sim_command(gpu,'run_training.py',args),protocol['training_child_seconds'] if mode=='train' else protocol['validation_child_seconds'],True,mode=='train')
    def audit_eval(gpu,label):
        job(gpu,label+'_audit',[str(PYTHON),str(HERE/'audit.py'),'evaluate',str(OUT/label)],600)
    def paired(fn):
        def wrapped(gpu):
            try:fn(gpu)
            except BaseException as e:
                with mutex:failures.append(dict(gpu=gpu,error=repr(e),traceback=traceback.format_exc()))
                stop.set();raise
        with ThreadPoolExecutor(max_workers=2) as pool:
            fs=[pool.submit(wrapped,g) for g in [0,1]]
            for f in fs:f.result()
    try:
        freeze();write(OUT/'receipt.json',dict(pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            protocol_sha256=sha(HERE/'protocol.json'),freeze_sha256=sha(HERE/'freeze.json'),finite_dependency_job=True,automatic_retries=0))
        publish()
        while True:
            assert time.time()<protocol['comparison_wait_deadline_unix'],'COMPARISON_WAIT_DEADLINE'
            status=read(OLD/'parallel_status.json')
            results=read(OLD/'combined_development_comparison.json')
            if dependency_ready(status,results):break
            time.sleep(30);publish()
        # Each final evaluation and optimizer audit must be present and passed.
        for arm in ['original','revised']:
            for seed in [63111,63117]:
                for label in [f'train_{arm}_{seed}_128']+[f'eval_{arm}_{seed}_128_{b}' for b in ['known_old','known_recent','known_transfer']]:
                    assert read(OLD/label/'independent_audit.json')['passed'],label
        for path in [OLD/'transition.lock',DATA/'contact_ppo_scientific_owner.lock']:
            lock=path.open('a')
            for attempt in range(12):
                try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);break
                except BlockingIOError:
                    if attempt==11:raise
                    time.sleep(5)
            lockfiles.append(lock)
        # Lock release alone is insufficient: reject any remaining old workers.
        ps=subprocess.check_output(['ps','-eo','args='],text=True)
        assert 'run_contact_learning_ppo_v4.py' not in ps,'COMPARISON_WORKER_STILL_LIVE'
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True)
        assert not apps.strip(),'GPU_ALREADY_OCCUPIED'
        freeze();deadline=protocol['execution_deadline_unix'];assert time.time()<deadline,'TOTAL_EXECUTION_WALL_CAP';state='validating_demonstrations';publish()
        write(OUT/'dependency_completed.json',dict(status=status,results_sha256=sha(OLD/'combined_development_comparison.json'),released_locks_acquired=True,execution_deadline_unix=deadline))
        def captures(gpu):
            names=['known_old','known_recent','known_transfer'] if gpu==0 else ['leaning']
            for name in names:
                label='capture_'+name
                job(gpu,label,sim_command(gpu,'capture_teacher.py',['--job',str(HERE/'jobs'/f'{name}.json')]),protocol['validation_child_seconds'],True)
        paired(captures)
        job(0,'build_dataset',[str(PYTHON),str(HERE/'build_dataset.py'),'--output',str(OUT/'dataset'),'--captures',*[str(OUT/('capture_'+n)) for n in ['known_old','known_recent','known_transfer','leaning']]],900)
        assert read(OUT/'dataset/dataset_audit.json')['passed']
        state='pretraining_and_reset_validation';publish()
        def prepare(gpu):
            task=['sitting','leaning'][gpu]
            trainer(gpu,'preflight_'+task,task,'preflight',64101)
            for seed in protocol['seeds']:
                label=f'bc_{task}_{seed}'
                job(gpu,label,[str(PYTHON),str(HERE/'pretrain.py'),'--bank',str(OUT/'dataset'/task),'--output',str(OUT/label),'--seed',str(seed)],1800)
            label='teacher_replay_'+task;trainer(gpu,label,task,'teacher',64101);audit_eval(gpu,label)
            # Full reward replay must succeed before any new PPO optimization.
        paired(prepare)
        state='ppo_smoke_checks';publish()
        def smoke(gpu):
            task=['sitting','leaning'][gpu]
            label='smoke_'+task;trainer(gpu,label,task,'train',64111,OUT/f'bc_{task}_64111/initial.pt',2,OUT/f'preflight_{task}/reset_audit.json')
            assert read(OUT/label/'final_audit.json')['passed']
        paired(smoke)
        state='training_and_evaluating';publish()
        def train(gpu):
            task=['sitting','leaning'][gpu]
            for seed in protocol['seeds']:
                checkpoint=OUT/f'bc_{task}_{seed}/initial.pt'; reset=OUT/f'preflight_{task}/reset_audit.json'
                label=f'eval_{task}_{seed}_000';trainer(gpu,label,task,'evaluate',seed,checkpoint);audit_eval(gpu,label)
                label=f'train_{task}_{seed}';trainer(gpu,label,task,'train',seed,checkpoint,256,reset)
                for stage in [128,256]:
                    ck=OUT/label/f'update_{stage:03d}.pt';e=f'eval_{task}_{seed}_{stage:03d}'
                    trainer(gpu,e,task,'evaluate',seed,ck);audit_eval(gpu,e)
        paired(train)
        baseline=read(HERE/'protected_sitting_cases.json'); protected=set(baseline['protected']); failed=set(baseline['failed']);report=[]
        for task in ['sitting','leaning']:
            meta=read(OUT/'dataset'/task/'metadata.json')
            if task=='leaning':
                ref=read(OUT/f'teacher_replay_{task}/independent_audit.json');protected_task={v for v,clean in zip(meta['ids'],ref['clean']) if clean};failed_task=set(meta['ids'])-protected_task
            else:protected_task=protected;failed_task=failed
            for seed in protocol['seeds']:
                for stage in [0,128,256]:
                    root=OUT/f'eval_{task}_{seed}_{stage:03d}';a=read(root/'independent_audit.json');assert a['passed']
                    clean={v for v,ok in zip(meta['ids'],a['clean']) if ok}
                    report.append(dict(task=task,seed=seed,updates=stage,clean=len(clean),total=len(meta['ids']),
                        repairs=sorted(clean&failed_task),regressions=sorted(protected_task-clean),audit=str(root/'independent_audit.json')))
        write(OUT/'comparison.json',dict(results=report,promoted=False,fresh_holdout_opened=False,success_criteria_unchanged=True,
            leaning_scope='Rear-trunk support on the18 existing development surfaces; not arbitrary hand or wall skills'))
        state='complete';publish(output_bytes=resources(),report=str(OUT/'comparison.json'))
    except BaseException as e:
        stop.set();state='stopped';failures.append(dict(error=repr(e),traceback=traceback.format_exc()));publish(retry=False);raise
    finally:
        for lock in lockfiles:lock.close()


if __name__=='__main__':main()
