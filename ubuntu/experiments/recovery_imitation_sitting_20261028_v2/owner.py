"""Single finite 4090-only owner; stops before PPO if complete-rollout gates fail."""
import fcntl
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import traceback
from common import ROOT,DATA,read,write,sha

HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/recovery_imitation_sitting_20261028_v2'
OLD=ROOT/'outputs/full_body_contact_imitation_v5'
V4=ROOT/'outputs/full_body_contact_imitation_v4'
BANK=V4/'dataset/sitting'
PY=ROOT/'IsaacLab/.venv/bin/python'
UUID='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'


def command_arguments(args):
    return list(map(str,args))


def main():
    protocol=read(HERE/'protocol.json');bindings=read(HERE/'freeze.json')
    OUT.mkdir(exist_ok=True);lock=(OUT/'owner.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not (OUT/'receipt.json').exists(),'NO_AUTOMATIC_RETRY'
    start=time.monotonic();deadline=start+min(protocol['wall_seconds'],protocol['execution_deadline_unix']-time.time());active=None;completed=[];records=[];locks=[];state='preflight'
    def publish(**extra):
        write(OUT/'status.json',dict(state=state,owner_pid=os.getpid(),active=active,completed=completed,
             seconds=time.monotonic()-start,gpu='RTX4090',gpu1_reserved='AIWorker',recurring_automation=False,
             physical_repair_claimed=False,**extra))
    def frozen():
        for path,digest in bindings.items():assert sha(path)==digest,('SOURCE_OR_INPUT_CHANGED',path)
    def resources():
        assert time.monotonic()<deadline,'WHOLE_RUN_WALL_LIMIT'
        assert shutil.disk_usage(OUT).free>8*1024**3,'DISK_RESERVE'
        size=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file());assert size<16*1024**3,'OUTPUT_LIMIT'
        mem=next(int(l.split()[1])*1024 for l in Path('/proc/meminfo').read_text().splitlines() if l.startswith('MemAvailable:'))
        assert mem>8*1024**3,'RAM_RESERVE'
        gpu=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-gpu=memory.used,temperature.gpu','--format=csv,noheader,nounits'],text=True)
        used,temp=map(int,gpu.strip().split(','));assert used<23552 and temp<85,('4090_RESOURCE_LIMIT',used,temp)
        if active:
            ps=subprocess.check_output(['ps','-eo','pgid=,rss='],text=True)
            rss=sum(int(l.split()[1])*1024 for l in ps.splitlines() if int(l.split()[0])==active['pid'])
            assert rss<48*1024**3,'OWNED_RSS_LIMIT'
        return size
    def gpu_check(pid,required=True):
        groups={int(l.split()[0]):int(l.split()[1]) for l in subprocess.check_output(['ps','-eo','pid=,pgid='],text=True).splitlines()}
        apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader'],text=True)
        found=[l.split(',')[0].strip() for l in apps.splitlines() if groups.get(int(l.split(',')[1]))==pid]
        assert (found or not required) and (not found or set(found)=={UUID}),('WRONG_GPU',found)
        return found
    def job(label,args,cap,gpu=False,training=False):
        nonlocal active
        args=command_arguments(args)
        frozen();resources();write(OUT/(label+'.command.json'),args)
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=UUID if gpu else '',CUDA_DEVICE_ORDER='PCI_BUS_ID',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
        checked=False;verified=False;t=time.monotonic()
        with (OUT/(label+'.log')).open('x') as log:
            child=subprocess.Popen(list(map(str,args)),cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            active=dict(label=label,pid=child.pid);publish()
            try:
                while child.poll() is None:
                    time.sleep(5);resources();assert time.monotonic()-t<cap,('CHILD_WALL_LIMIT',label)
                    if gpu and not checked:
                        devices=gpu_check(child.pid,required=time.monotonic()-t>90)
                        if devices:
                            checked=True
                            write(OUT/(label+'_gpu.json'),dict(pid=child.pid,devices=devices,passed=True))
                    if training and not verified and (OUT/label/'weights_ready.json').exists():
                        gpu_check(child.pid)
                        from audit import weights
                        result=weights(OUT/label);verified=True;write(OUT/(label+'_verified.json'),result)
                    publish()
                code=child.returncode
            except BaseException:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
                raise
            finally:
                write(OUT/(label+'.exit.json'),dict(returncode=child.returncode,seconds=time.monotonic()-t,verified_weight_update=verified))
                active=None
        assert code==0,(label,code)
        if gpu:assert checked,'GPU_DEVICE_NOT_VERIFIED'
        if training:assert verified,'NO_LIVE_WEIGHT_UPDATE_VERIFICATION'
        completed.append(label);frozen();publish()
    def sim(label,mode,seed,ck=None,recovery_round=None,demo=None):
        args=['bash',ROOT/'scripts/run_supported_gpu0.sh',HERE/'run_training.py','--bank',BANK,'--output',OUT/label,
              '--mode',mode,'--seed',seed,'--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
        if ck:args+=['--checkpoint',ck]
        if recovery_round is not None:args+=['--recovery-round',recovery_round]
        if mode=='train':args+=['--updates',256,'--reset-audit',V4/'preflight_sitting/reset_audit.json','--demonstrations',demo]
        job(label,args,43200 if mode=='train' else 7200,True,mode=='train')
        if mode!='train':job(label+'_audit',[PY,HERE/'audit.py','evaluate',OUT/label],600)
    def result(label,seed,stage):
        audit=read(OUT/label/'independent_audit.json');assert audit['passed']
        rows=read(OUT/label/'results.json')['episodes'];baseline=read(HERE/'protected_sitting_cases.json')
        clean={r['id'] for r in rows if r['clean']};protected=set(baseline['protected']);failed=set(baseline['failed'])
        return dict(seed=seed,stage=stage,label=label,clean=len(clean),repairs=sorted(clean&failed),regressions=sorted(protected-clean),
                    complete_trials=len(rows),duration_s=20,independent_audit_passed=True)
    try:
        frozen();assert read(HERE/'construction_audit.json')['passed'];assert read(HERE/'source_review.json')['passed']
        assert read(OLD/'status.json')['state']=='complete' and not read(OLD/'status.json')['active']
        for task in ['sitting','leaning']:
            for seed in [64111,64117]:
                assert read(OLD/f'train_{task}_{seed}/final_audit.json')['passed']
                for u in [0,128,256]:assert read(OLD/f'eval_{task}_{seed}_{u:03d}/independent_audit.json')['passed']
        assert read(V4/'preflight_sitting/reset_audit.json')['passed']
        assert read(V4/'teacher_replay_sitting/independent_audit.json')['clean_count']==64
        for path in [DATA/'contact_parallel_gpu_v1/transition.lock',DATA/'contact_ppo_scientific_owner.lock']:
            f=path.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
        apps=subprocess.check_output(['nvidia-smi','-i',UUID,'--query-compute-apps=pid','--format=csv,noheader'],text=True)
        assert not apps.strip(),('4090_BUSY',apps)
        write(OUT/'receipt.json',dict(owner_pid=os.getpid(),started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
            protocol_sha256=sha(HERE/'protocol.json'),freeze_sha256=sha(HERE/'freeze.json'),prior_comparison=sha(OLD/'comparison.json'),
            automatic_retries=0,one_scientific_owner=True,wall_seconds=protocol['wall_seconds']))
        state='cloning_and_recovery';publish();finals={}
        for seed in protocol['seeds']:
            previous=None
            for round in [0,1,2]:
                label=f'clone_{seed}_{round}'
                args=[PY,HERE/'clone.py','--bank',BANK,'--output',OUT/label,'--seed',seed,'--round',round]
                if round:
                    rec=f'recovery_{seed}_{round-1}';sim(rec,'recovery',seed,OUT/previous/'initial.pt',round-1)
                    args+=['--checkpoint',OUT/previous/'initial.pt','--prior-demo',OUT/previous/'demonstrations.npz','--recovery',OUT/rec]
                job(label,args,3600,True);previous=label
            ck=OUT/previous/'initial.pt';ev=f'clone_evaluation_{seed}';sim(ev,'evaluate',seed,ck)
            r=result(ev,seed,'clone');records.append(r);write(OUT/'comparison.json',dict(results=records,complete=False))
            finals[seed]=(ck,OUT/previous/'demonstrations.npz')
        write(OUT/'closed_loop_gate.json',dict(passed=all(not r['regressions'] for r in records),results=records,
            requires_all64_named_protected_cases=True,teacher_interventions=False))
        assert all(not r['regressions'] for r in records),'COMPLETE_SITTING_CLONE_GATE_FAILED_PPO_NOT_STARTED'
        state='ppo_with_imitation';publish()
        for seed in protocol['seeds']:
            ck,demo=finals[seed];tr=f'train_sitting_{seed}';sim(tr,'train',seed,ck,demo=demo)
            for u in [128,256]:
                ev=f'eval_sitting_{seed}_{u}';sim(ev,'evaluate',seed,OUT/tr/f'update_{u:03d}.pt')
                records.append(result(ev,seed,u));write(OUT/'comparison.json',dict(results=records,complete=False))
        state='complete';write(OUT/'comparison.json',dict(results=records,complete=True,automatic_promotion=False));publish(output_bytes=resources())
    except BaseException as e:
        state='stopped';write(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),completed=completed,seconds=time.monotonic()-start))
        publish(error=repr(e));raise

if __name__=='__main__':main()
