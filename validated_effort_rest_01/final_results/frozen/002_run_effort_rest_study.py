"""Frozen two-seed direct-effort PPO study, with persistent matched evaluation."""
import datetime,hashlib,json,os,shutil,subprocess,traceback
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting')
STORE=Path('/data/humanoid-sitting-perception')
PREF=STORE/'effort_rest_01';OUT=PREF/'ppo'
BANK=STORE/'supported_surfaces_01/transition_banks_03'
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def status(phase,**kw):
    payload=dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),study=str(OUT),**kw)
    for p in [OUT/'status.json',PREF/'status.json']:
        tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(payload,indent=2));tmp.replace(p)

assert read(PREF/'fixed_train/results.json')['successes']==13
assert read(PREF/'fixed_holdout/results.json')['successes']==16
assert read(PREF/'reset_probe/reset_probe.json')['passed']
smoke=read(PREF/'optimizer_smoke_02/verification.json')
assert smoke['updates']==2 and smoke['transitions']==3328 and smoke['finite']
smoke_eval=read(PREF/'smoke_evaluation_02/results.json')
assert smoke_eval['successes']==13,('The two-update deterministic policy did not retain all13 supported states',smoke_eval['successes'])
OUT.mkdir(exist_ok=False);(OUT/'frozen').mkdir()
old=read(STORE/'supported_surfaces_01/transition_ppo_01/protocol.json')['hashes']
old.update(read(STORE/'resting_bounded_ppo_02/frozen/manifest.json'))
assert all(sha(Path(p))==h for p,h in old.items())
paths=[ROOT/'scripts/run_effort_rest.py',ROOT/'scripts/test_effort_rest.py',Path(__file__).resolve(),ROOT/'docs/EFFORT_REST_PROTOCOL.md',
       ROOT/'sitting_task/src/humanoid_sitting/effort_rest.py']
paths+=list(map(Path,read(PREF/'smoke_evaluation_02/manifest.json')['source_hashes']))
paths+=list(BANK.glob('*/*.npz'))+list(BANK.glob('*/*.json'))
for label in ['fixed_train','fixed_holdout','reset_probe','exploration_probe','exploration_probe_02','optimizer_smoke_02','smoke_evaluation_02']:
    paths += [x for x in (PREF/label).glob('*') if x.is_file() and x.suffix in ['.json','.npz']]
paths=list(dict.fromkeys(x.resolve() for x in paths))
frozen={str(p):sha(p) for p in paths}
protocol=dict(seeds=[54511,54517],updates_per_seed=256,steps_per_update=64,environments=26,
    transitions_per_seed=425984,total_transitions=851968,development_smoke_transitions=3328,
    initial_global_settle_excluded_s=.5,indexed_reset_warmup_ticks=25,seconds=20,
    train_geometries=13,heldout_geometries=16,actor_critic_sensor_channels=402,
    initial_std=.15,learning_rate=1.e-4,adaptive_KL=.005,epochs=5,minibatches=4,entropy=.001,gamma=.998,lam=.97,
    primary_objective='minimize total squared physical Nm over all37 actuated joints, subject to valid stable support',
    primary_comparison='whole20second and final10second means on common successful matched trials; failures reported first',
    calibration=str(PREF/'fixed_train/results.json'),fixed_baselines={b:str(PREF/f'fixed_{b}') for b in ['train','holdout']},
    smoke_success=smoke_eval['successes'],prior_study_hashes=old,hashes=frozen,
    preflight_note='Baseline sources predate adding initial_std CLI option; fixed behavior is unchanged and those source snapshots/manifests are preserved. Random exploratory contacts retain physical failure labels.',
    no_policy_promotion=True)
(OUT/'protocol.json').write_text(json.dumps(protocol,indent=2))
for i,path in enumerate(paths):shutil.copy2(path,OUT/'frozen'/f'{i:03d}_{path.name}')
(OUT/'frozen/manifest.json').write_text(json.dumps(frozen,indent=2))
def check():
    assert all(sha(Path(p))==h for p,h in frozen.items()),'Frozen experiment changed'
    assert all(sha(Path(p))==h for p,h in old.items()),'Prior experiment changed'
def command(name,mode,bank='train',gpu=0,seed=54501,n=None,**extra):
    cmd=['bash','scripts/run_supported_gpu0.sh' if gpu==0 else 'scripts/run_resting_gpu1.sh','python','-u','../scripts/run_effort_rest.py',
         '--mode',mode,'--bank',str(BANK/bank),'--output',str(OUT/name),'--seed',str(seed),'--seconds','20','--device','cuda:0',
         '--calibration',str(PREF/f'fixed_{bank}/results.json'),f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']
    if n:cmd+=['--num-envs',str(n)]
    for k,v in extra.items():cmd+=['--'+k.replace('_','-'),str(v)]
    (OUT/f'{name}.command.json').write_text(json.dumps(cmd,indent=2));return cmd
workers=[]
try:
    check();status('training',seeds=protocol['seeds'])
    for gpu,seed in enumerate(protocol['seeds']):
        name=f'train_{seed}';log=(OUT/f'{name}.log').open('x')
        process=subprocess.Popen(command(name,'train',gpu=gpu,seed=seed,n=26,iterations=256),cwd=ROOT,
            env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
        workers.append((process,log,seed))
    (OUT/'workers.json').write_text(json.dumps([dict(pid=p.pid,seed=s) for p,l,s in workers],indent=2))
    # Both persistent children remain alive even if the other encounters an error.
    exits=[]
    for process,log,seed in workers:
        exits.append((seed,process.wait()));log.close()
    assert all(code==0 for seed,code in exits),exits
    check()
    for seed in protocol['seeds']:
        v=read(OUT/f'train_{seed}/verification.json')
        assert v['updates']==256 and v['active']+v['warmup']==v['transitions']==425984
        for bank in ['train','holdout']:
            name=f'eval_{seed}_{bank}';status('evaluating',job=name)
            with (OUT/f'{name}.log').open('x') as log:
                subprocess.run(command(name,'evaluate',bank=bank,checkpoint=OUT/f'train_{seed}/final.pt'),cwd=ROOT,
                    env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    check();status('complete_pending_verification')
except BaseException:
    status('failed',traceback=traceback.format_exc());raise
