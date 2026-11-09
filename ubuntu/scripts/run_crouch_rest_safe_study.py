"""Persistent bounded pilot worker; no controller tuning or extra training."""
import argparse,json,hashlib,time,traceback
from pathlib import Path
from run_progressive_seat_study import command as old_command
from run_transfer_assessment import run,write,status

ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01')
STUDY=DATA/'ppo_safe_surfaces_01';SEEDS=[58111,58117];ELIGIBLE=[0,1,2,4,5,8]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(protocol):
    for group in ['hashes','prior_hashes']:
        for p,h in protocol[group].items():assert sha(p)==h,p

def command(folder,gpu,label,mode,start=15,seed=58101,checkpoint=None,updates=None,warmup=0,holdout=False,kind='new'):
    cmd=old_command(folder,gpu,label,mode,stage=0,seed=seed,checkpoint=checkpoint,updates=updates,warmup=warmup,kind=kind)
    if kind=='new':cmd[2]='../scripts/run_crouch_rest_audited_ppo.py';cmd+=['--learning-start',str(start)]
    if holdout:
        cmd[cmd.index('--bank')+1]=str(DATA/'reserved_bank_01');cmd[cmd.index('--num-envs')+1]='16'
        if '--reset-library' in cmd:
            i=cmd.index('--reset-library');del cmd[i:i+2]
    return cmd

def gate(folder):
    rows=read(folder/'results.json')['episodes'];assert len(rows)==18 and all(x['steps']==1000 for x in rows)
    included=[x for x in rows if x['geometry_index'] in ELIGIBLE];assert len(included)==12
    excluded=[x for x in rows if x['geometry_index'] not in ELIGIBLE]
    return dict(passed=all(x['strict_transfer'] for x in included),included_stable=sum(x['success'] for x in included),included_strict=sum(x['strict_transfer'] for x in included),included_physical=sum(any(x['failures'].values()) for x in included),included_trials=12,excluded_stable=sum(x['success'] for x in excluded),excluded_physical=sum(any(x['failures'].values()) for x in excluded),excluded_trials=6)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
    protocol=read(STUDY/'protocol.json');check(protocol);folder=STUDY/f'seed_{a.seed}';folder.mkdir(exist_ok=False);state=STUDY/f'status_{a.seed}.json';start=15;cp=None;records=[];total=0
    try:
        for block,updates in enumerate(protocol['block_updates']):
            check(protocol);label=f'block_{block:02d}_control{start}s';status(state,'training',block=block,learning_start_s=start,completed_updates=total,active=label)
            run(folder,label,command(folder,a.gpu,label,'train',start,a.seed,cp,updates,8 if block==0 else 0))
            cp=folder/label/'final.pt';v=read(folder/label/'verification.json');assert v['updates']==updates and v['transitions']==updates*32*5*18;total+=updates
            label=f'gate_{block:02d}_control{start}s';status(state,'current_window_validation',learning_start_s=start,completed_updates=total,active=label)
            run(folder,label,command(folder,a.gpu,label,'evaluate',start,checkpoint=cp));current=gate(folder/label)
            next_start=start;earlier=None;proposal={15:13,13:11,11:10,10:10}[start]
            if block<3 and current['passed'] and proposal<start:
                next_label=f'advance_{block:02d}_control{proposal}s';status(state,'earlier_window_validation',learning_start_s=proposal,completed_updates=total,active=next_label)
                run(folder,next_label,command(folder,a.gpu,next_label,'evaluate',proposal,checkpoint=cp));earlier=gate(folder/next_label)
                if earlier['passed']:next_start=proposal
            records.append(dict(block=block,updates=updates,learning_start_s=start,checkpoint=str(cp),checkpoint_sha256=sha(cp),evaluation=str(folder/label),current_gate=current,earlier_gate=earlier,next_start_s=next_start));write(folder/'curriculum.json',records);start=next_start
        assert total==64
        finished=dict(checkpoint=str(cp),sha256=sha(cp),updates=total,timestamp=time.time(),learning_start_s=records[-1]['learning_start_s'],final_development_evaluation=records[-1]['evaluation'])
        write(STUDY/f'training_finished_{a.seed}.json',finished);status(state,'waiting_for_other_final',**finished)
        while not all((STUDY/f'training_finished_{s}.json').exists() for s in SEEDS):
            for s in SEEDS:
                f=STUDY/f'status_{s}.json'
                assert not f.exists() or read(f).get('phase')!='failed',f
            time.sleep(15)
        barrier=STUDY/'holdout_barrier.json'
        if a.gpu==0:write(barrier,dict(timestamp=time.time(),selection='Full-budget final checkpoints; learned control-window timing frozen by development gates',finals={str(s):read(STUDY/f'training_finished_{s}.json') for s in SEEDS}))
        while not barrier.exists():time.sleep(5)
        old=DATA.parent/'contact_effort_repair_01'
        jobs=[('current_reference','reference','current_reference',None),('bridge_reference','reference','new',None)] if a.gpu==0 else [(f'current_{s}','evaluate','current_policy',old/f'sitting_ppo_01/train_{s}/final.pt') for s in [56211,56217]]
        jobs.append((f'policy_{a.seed}','evaluate','new',cp))
        for name,mode,kind,checkpoint in jobs:
            label='holdout_'+name;status(state,'holdout_evaluation',active=label)
            run(STUDY,label,command(STUDY,a.gpu,label,mode,finished['learning_start_s'],checkpoint=checkpoint,holdout=True,kind=kind))
        check(protocol);status(state,'complete',**finished)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
