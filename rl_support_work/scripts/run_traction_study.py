"""Bounded two-seed comparison; requires a frozen, admitted preflight protocol."""
import argparse,json,hashlib,time,traceback
from pathlib import Path
from run_crouch_rest_safe_study import command as previous_command
from run_transfer_assessment import run,write,status
ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/traction_transfer_01')
STUDY=DATA/'ppo_01';SEEDS=[58311,58317]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p):
    for group in ['hashes','prior_hashes']:
        for name,h in p[group].items():assert sha(name)==h,name
def command(folder,gpu,label,mode,seed=58301,checkpoint=None,updates=None,warmup=0,holdout=False,kind='new',probe=False):
    inherited='new' if kind in ['new','bridge'] else kind
    cmd=previous_command(folder,gpu,label,mode,start=11 if kind=='new' else 15,seed=seed,checkpoint=checkpoint,updates=updates,warmup=warmup,kind=inherited)
    if kind=='new':
        cmd[2]='../scripts/run_traction_transfer_ppo.py';cmd+=['--traction-config',str(DATA/'selected_controller.json'),'--eligible-indices','0,1,2,3,4,5,6,7,8']
    if holdout:
        cmd[cmd.index('--bank')+1]=str(DATA/'reserved_bank_01');cmd[cmd.index('--num-envs')+1]='16'
        if '--reset-library' in cmd:
            i=cmd.index('--reset-library');del cmd[i:i+2]
    if probe:cmd+=['--reset-probe']
    return cmd
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
    protocol=read(STUDY/'protocol.json');assert protocol['admitted'];check(protocol)
    folder=STUDY/f'seed_{a.seed}';folder.mkdir(exist_ok=False);state=STUDY/f'status_{a.seed}.json'
    try:
        status(state,'training',seed=a.seed,updates=64,control_start_s=11)
        run(folder,'train',command(folder,a.gpu,'train','train',a.seed,updates=64,warmup=8))
        cp=folder/'train/final.pt';v=read(folder/'train/verification.json');assert v['updates']==64 and v['transitions']==184320
        final=dict(checkpoint=str(cp),sha256=sha(cp),timestamp=time.time(),updates=64,control_start_s=11)
        write(STUDY/f'training_finished_{a.seed}.json',final)
        status(state,'development_evaluation',**final)
        run(folder,'development_final',command(folder,a.gpu,'development_final','evaluate',checkpoint=cp))
        status(state,'waiting_for_both_final_checkpoints',**final)
        while not all((STUDY/f'training_finished_{s}.json').exists() for s in SEEDS):
            for s in SEEDS:
                f=STUDY/f'status_{s}.json';assert not f.exists() or read(f).get('phase')!='failed',f
            time.sleep(15)
        barrier=STUDY/'holdout_barrier.json'
        if a.gpu==0:write(barrier,dict(timestamp=time.time(),selection='Final full-budget checkpoints only; no holdout tuning or selection',finals={str(s):read(STUDY/f'training_finished_{s}.json') for s in SEEDS}))
        while not barrier.exists():time.sleep(5)
        for job in protocol['holdout_jobs'][str(a.gpu)]:
            label='holdout_'+job['name'];status(state,'holdout_evaluation',active=label)
            run(STUDY,label,command(STUDY,a.gpu,label,job['mode'],checkpoint=job.get('checkpoint'),holdout=True,kind=job['kind']))
        label=f'holdout_policy_{a.seed}';status(state,'holdout_evaluation',active=label)
        run(STUDY,label,command(STUDY,a.gpu,label,'evaluate',checkpoint=cp,holdout=True))
        check(protocol);status(state,'complete',**final)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
