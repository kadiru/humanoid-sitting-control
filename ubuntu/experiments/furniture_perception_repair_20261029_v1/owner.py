"""One finite4090 continuation. Wait for existing PPO; never interrupts it."""
import fcntl,os,shutil,signal,subprocess,time,traceback
from pathlib import Path
from paths import ROOT,HERE,OUT,ASSETS,PRIOR,PPO,BODY,CHECK,GPU,PY,read,write,sha
DATA=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(')')[-1].split()[0]!='Z'
    except FileNotFoundError:return False

def main():
    limits=read(HERE/'protocol.json')['resources'];bindings=read(HERE/'freeze.json')
    OUT.mkdir(exist_ok=True);own=(OUT/'owner.lock').open('a');fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'receipt.json').exists(),'NO_AUTOMATIC_RETRY'
    start=time.monotonic();execution=None;state='waiting_for_ppo';active=None;completed=[];locks=[];launches=0
    def publish(**extra):
        write(OUT/'status.json',dict(state=state,owner_pid=os.getpid(),active=active,completed=completed,
            elapsed_s=time.monotonic()-start,execution_s=None if execution is None else time.monotonic()-execution,
            simulator_launches=launches,gpu='4090only',gpu4070='AIWorker untouched',recurring_automation=False,**extra))
    def frozen():
        for p,h in bindings.items():assert sha(p)==h,('FROZEN_SOURCE_OR_INPUT_CHANGED',p)
        if (OUT/'policy_bindings.json').exists():
            for p,h in read(OUT/'policy_bindings.json').items():assert sha(p)==h,('POLICY_CHANGED',p)
    def resource():
        assert time.monotonic()-start<limits['wait_seconds']+limits['execution_seconds']
        if execution is not None:assert time.monotonic()-execution<limits['execution_seconds']
        assert shutil.disk_usage(OUT).free>limits['free_disk_bytes']
        assert sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())<limits['output_bytes']
        mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        assert mem>limits['free_ram_bytes']
        if active:
            rows=subprocess.check_output(['ps','-eo','pgid=,rss='],text=True).splitlines()
            assert sum(int(l.split()[1])*1024 for l in rows if int(l.split()[0])==active['pid'])<limits['owned_rss_bytes']
    def apps():return subprocess.check_output(['nvidia-smi','-i',GPU,'--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
    def job(label,argv,gpu=False):
        nonlocal active
        resource();frozen()
        for _ in range(12):
            if not apps():break
            time.sleep(5)
        assert not apps(),'4090_BUSY_BY_ANOTHER_OWNER'
        argv=list(map(str,argv));write(OUT/(label+'.command.json'),argv)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=GPU if gpu else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        t=time.monotonic();verified=not gpu
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active=dict(label=label,pid=child.pid);publish()
            try:
                while child.poll() is None:
                    time.sleep(5);resource();assert time.monotonic()-t<limits['child_seconds'],('CHILD_TIMEOUT',label)
                    if gpu:
                        groups={int(l.split()[0]):int(l.split()[1]) for l in subprocess.check_output(['ps','-eo','pid=,pgid='],text=True).splitlines()}
                        query=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
                        devices={l.split(',')[0].strip() for l in query.splitlines() if groups.get(int(l.split(',')[1]))==child.pid}
                        assert not devices or devices=={GPU},('WRONG_GPU',devices);verified|=bool(devices)
                        temp,mem=map(int,subprocess.check_output(['nvidia-smi','-i',GPU,'--query-gpu=temperature.gpu,memory.used','--format=csv,noheader,nounits'],text=True).split(','))
                        assert temp<limits['gpu_temperature_C'] and mem<limits['gpu_memory_MiB']
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
                write(OUT/(label+'.exit.json'),dict(returncode=child.returncode,seconds=time.monotonic()-t,gpu_verified=verified));active=None
        completed.append(label);publish()
    def cpu(label,script,args=()):job(label,[PY,script,*args])
    def simulation(label,script,args):
        nonlocal launches
        launches+=1;assert launches<=limits['maximum_simulator_launches']
        job(label,['bash',ROOT/'scripts/run_supported_gpu0.sh',script,*args,'--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false'],True)
    try:
        frozen();assert read(HERE/'source_review.json')['passed'] and read(HERE/'cpu_contract_checks.json')['passed']
        write(OUT/'receipt.json',dict(owner_pid=os.getpid(),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),protocol_sha256=sha(HERE/'protocol.json'),freeze_sha256=sha(HERE/'freeze.json'),recurring_automation=False))
        while True:
            resource();dep=read(PPO/'status.json');publish(dependency=dep)
            assert dep['state']!='stopped',('PPO_DEPENDENCY_FAILED',dep)
            if dep['state']=='complete' and dep['active'] is None and not alive(dep['owner_pid']):break
            assert time.monotonic()-start<limits['wait_seconds'],'DEPENDENCY_WAIT_TIMEOUT'
            time.sleep(30)
        comparison=read(PPO/'comparison.json');assert comparison['complete']
        for seed in [64111,64117]:
            assert read(PPO/f'train_sitting_{seed}/status.json')['state']=='complete'
            for update in [128,256]:assert read(PPO/f'eval_{seed}_{update}/independent_audit.json')['passed']
        for p in [DATA/'contact_parallel_gpu_v1/transition.lock',DATA/'contact_ppo_scientific_owner.lock']:
            f=p.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        execution=time.monotonic();assert not apps(),'4090_NOT_IDLE'
        rows=[r for r in comparison['results'] if r['label'] in ['eval_64111_256','eval_64117_256']];assert len(rows)==2
        selected=min(rows,key=lambda r:(len(r['regressions']),-r['clean'],r['seed']))
        checkpoint=PPO/f"train_sitting_{selected['seed']}/update_256.pt"
        leaning=ROOT/'outputs/full_body_contact_imitation_v5/train_leaning_64111/update_256.pt'
        write(OUT/'policy_selection.json',dict(selected_original_result=selected,selection_rule=read(HERE/'protocol.json')['policy_binding'],automatic_promotion=False))
        write(OUT/'policy_bindings.json',{str(p):sha(p) for p in [checkpoint,leaning]})
        state='retention72';publish()
        simulation('retention72',HERE/'retention.py',['--mode','evaluate','--bank',ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting','--output',OUT/'retention72','--checkpoint',checkpoint,'--seed',selected['seed']])
        cpu('retention72_audit',PRIOR/'audit.py',['evaluate',OUT/'retention72'])
        state='furniture_rollouts';publish()
        for c in read(ASSETS/'cases.json'):
            label='robot_'+c['id']
            simulation(label,HERE/'robot_trial.py',['--case',c['id'],'--output',OUT/label,'--checkpoint',checkpoint if c['task']=='sitting' else leaning])
            cpu(label+'_depth_audit',HERE/'audit_depth.py',['--case',c['id'],'--output',OUT/label])
        cpu('plan_observed_probes',HERE/'plan_probes.py')
        if read(OUT/'observed_probe_points.json'):
            state='body_support';publish()
            simulation('body_support',HERE/'body_support_probe.py',['--output',OUT/'body_support','--inventory',BODY/'body_colliders.json'])
            cpu('body_support_audit',HERE/'audit_support.py',['--output',OUT/'body_support'])
        # Account actual physics/control work across every completed launch.
        robot_results=[read(OUT/'retention72/results.json')]+[read(OUT/('robot_'+c['id'])/'results.json') for c in read(ASSETS/'cases.json')]
        physical=sum(x['actual_total_physics_steps'] for x in robot_results)
        controls=sum(x['active_transitions']+x['warmup_transitions'] for x in robot_results)
        if (OUT/'body_support/result.json').exists():physical+=read(OUT/'body_support/result.json')['physical_steps']
        assert physical<=limits['maximum_physics_steps'] and controls<=limits['maximum_control_instance_steps']
        write(OUT/'resource_accounting.json',dict(physics_steps=physical,control_instance_steps=controls,launches=launches,wait_seconds=execution-start,execution_seconds_before_media=time.monotonic()-execution))
        state='rendering';publish();cpu('report',HERE/'report.py');cpu('video',HERE/'render_review.py')
        state='complete';publish()
    except BaseException as e:
        state='stopped';write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=completed));publish(error=repr(e));raise
if __name__=='__main__':main()
