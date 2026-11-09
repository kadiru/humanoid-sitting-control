"""Persistent validation of the new direct-effort experiment, no full study yet."""
import concurrent.futures,datetime,json,os,subprocess,traceback
from pathlib import Path
ROOT=Path('/home/kadir/humanoid-sitting');OUT=Path('/data/humanoid-sitting-perception/effort_rest_01')
BANK=Path('/data/humanoid-sitting-perception/supported_surfaces_01/transition_banks_03')
OUT.mkdir(exist_ok=False)
def status(phase,**kw):
    p=OUT/'status.tmp';p.write_text(json.dumps(dict(phase=phase,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2));p.replace(OUT/'status.json')
def sim(name,mode,bank='train',gpu=0,n=None,seed=54501,**extra):
    cmd=['bash','scripts/run_supported_gpu0.sh' if gpu==0 else 'scripts/run_resting_gpu1.sh','python','-u','../scripts/run_effort_rest.py',
         '--mode',mode,'--bank',str(BANK/bank),'--output',str(OUT/name),'--seed',str(seed),'--seconds','20','--device','cuda:0',
         f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']
    if n:cmd+=['--num-envs',str(n)]
    for k,v in extra.items():
        cmd+=['--'+k.replace('_','-')]
        if v is not True:cmd+=[str(v)]
    (OUT/f'{name}.command.json').write_text(json.dumps(cmd,indent=2))
    with (OUT/f'{name}.log').open('x') as log:
        subprocess.run(cmd,cwd=ROOT,env=dict(os.environ,OPENBLAS_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,check=True)
    return json.loads((OUT/name/'results.json').read_text()) if (OUT/name/'results.json').exists() else None
try:
    status('preflight_running')
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        def probes():
            sim('reset_probe','probe',gpu=1,reset_probe=True)
            return sim('exploration_probe','probe',gpu=1,noise_std=.3)
        future=pool.submit(probes)
        fixed=sim('fixed_train','fixed');assert fixed['successes']==13,('fixed_train',fixed['successes'])
        heldout=sim('fixed_holdout','fixed',bank='holdout')
        noise=future.result()
    status('preflight_baselines_done',fixed_train=fixed['successes'],fixed_holdout=heldout['successes'],exploration=noise['successes'])
    assert all(not any(x['failures'].values()) for x in noise['episodes']),'Initial exploration has physical failures; inspect before full training'
    sim('optimizer_smoke','train',gpu=1,n=26,seed=54507,iterations=2,calibration=OUT/'fixed_train/results.json')
    smoke=sim('smoke_evaluation','evaluate',gpu=0,checkpoint=OUT/'optimizer_smoke/final.pt',calibration=OUT/'fixed_train/results.json')
    assert smoke['successes']==13,('smoke_eval',smoke['successes'])
    status('preflight_complete',fixed_train=fixed['successes'],fixed_holdout=heldout['successes'],exploration=noise['successes'],smoke_success=smoke['successes'])
except BaseException:
    status('preflight_failed',traceback=traceback.format_exc());raise
