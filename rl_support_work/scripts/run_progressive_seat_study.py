"""Persistent bounded PPO curriculum worker. This is a simulation job, not an agent."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
from run_transfer_assessment import run,write,status
ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/near_seat_curriculum_01')
OLD=DATA.parent/'contact_effort_repair_01'
SEEDS=[57111,57117]
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(protocol):
    for group in ['hashes','prior_hashes']:
        for p,h in protocol[group].items():assert sha(p)==h,p

def command(folder,gpu,label,mode,stage=0,seed=57101,checkpoint=None,updates=None,warmup=0,bank='development',kind='new',probe=False):
    script={'new':'run_progressive_seat_ppo.py','current_reference':'run_contact_effort_repair.py','current_policy':'run_stable_effort_sitting.py'}[kind]
    bankpath=DATA.parent/'seat_load_transfer_01/banks/development' if bank=='development' else DATA/'reserved_bank'
    cmd=['bash','scripts/'+('run_supported_gpu0.sh' if gpu==0 else 'run_resting_gpu1.sh'),'../scripts/'+script,'--bank',str(bankpath),'--output',str(folder/label),'--mode',mode,'--surface','sitting','--profiles',str(OLD/'sitting_repaired_profile.json'),'--plane-calibration',str(OLD/'plane_calibration.json'),'--seed',str(seed),'--start-stage','0','--num-envs','18' if bank=='development' else '16','--device','cuda:0',f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']
    if kind=='new':
        cmd+=['--curriculum-stage',str(stage)]
        if bank=='development':cmd+=['--reset-library',str(DATA/'reset_library_01.npz')]
    if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
    if mode=='train':cmd+=['--iterations',str(updates),'--critic-warmup',str(warmup)]
    if probe:cmd+=['--reset-probe']
    if kind=='current_policy':
        parent=read(DATA.parent/'split_support_curriculum_02/ppo_01/sitting/seed_55211.json')['last_checkpoint']
        cmd+=['--parent-checkpoint',parent,'--calibration',str(OLD/'sitting_preflight_01/calibration.json')]
    return cmd

def gate(path):
    rows=read(path/'results.json')['episodes'];assert len(rows)==18
    physical=sum(any(x['failures'].values()) for x in rows);stable=sum(x['success'] for x in rows);strict=sum(x['strict_transfer'] for x in rows)
    loaded=sum(x['metrics_last5s']['vertical_support_load_bw']>=.6 for x in rows)
    return dict(passed=physical==0 and stable==18 and strict==18 and loaded==18,physical=physical,stable=stable,strict=strict,seat_majority=loaded,trials=18)

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);p.add_argument('--seed',type=int,choices=SEEDS,required=True);a=p.parse_args()
    study=DATA/'ppo_01';folder=study/f'seed_{a.seed}';folder.mkdir(exist_ok=False);protocol=read(study/'protocol.json');check(protocol);state=study/f'status_{a.seed}.json';stage=5;cp=None;records=[];total=0
    try:
        for block,updates in enumerate(protocol['block_updates']):
            check(protocol);label=f'block_{block:02d}_stage{stage}';warm=8 if block==0 else 0
            status(state,'training',block=block,stage=stage,completed_updates=total,active=label)
            run(folder,label,command(folder,a.gpu,label,'train',stage,a.seed,cp,updates,warm))
            cp=folder/label/'final.pt';v=read(folder/label/'verification.json');assert v['updates']==updates and v['transitions']==updates*32*5*18;total+=updates
            evaluation=f'gate_{block:02d}_stage{stage}';status(state,'stage_evaluation',stage=stage,completed_updates=total,active=evaluation)
            run(folder,evaluation,command(folder,a.gpu,evaluation,'evaluate',stage,checkpoint=cp))
            decision=gate(folder/evaluation);next_stage=max(0,stage-1) if decision['passed'] else stage
            records.append(dict(block=block,updates=updates,stage=stage,checkpoint=str(cp),checkpoint_sha256=sha(cp),evaluation=str(folder/evaluation),gate=decision,next_stage=next_stage));write(folder/'curriculum.json',records);stage=next_stage
        assert total==128
        write(study/f'training_finished_{a.seed}.json',dict(checkpoint=str(cp),sha256=sha(cp),updates=total,time=time.time(),last_trained_stage=records[-1]['stage'],next_stage=stage))
        for s,label in [(0,'final_development_standing'),(5,'final_development_near')]:
            status(state,'final_development_evaluation',active=label)
            run(folder,label,command(folder,a.gpu,label,'evaluate',s,checkpoint=cp))
        status(state,'waiting_for_other_final_checkpoint')
        while not all((study/f'training_finished_{seed}.json').exists() for seed in SEEDS):time.sleep(10)
        barrier=study/'holdout_barrier.json'
        if a.gpu==0:write(barrier,dict(time=time.time(),selection='Full-budget finals only; curriculum gates use development only; no checkpoint selection',checkpoints={str(s):read(study/f'training_finished_{s}.json') for s in SEEDS}))
        while not barrier.exists():time.sleep(5)
        jobs=[('current_reference','reference','current_reference',None),('new_reference','reference','new',None)] if a.gpu==0 else [(f'current_{s}','evaluate','current_policy',OLD/f'sitting_ppo_01/train_{s}/final.pt') for s in [56211,56217]]
        jobs.append((f'policy_{a.seed}','evaluate','new',cp))
        for label,mode,kind,checkpoint in jobs:
            label='holdout_'+label;status(state,'holdout_evaluation',active=label)
            run(study,label,command(study,a.gpu,label,mode,0,checkpoint=checkpoint,bank='reserved',kind=kind))
        check(protocol);status(state,'complete',checkpoint=str(cp),updates=total,last_trained_stage=records[-1]['stage'])
    except Exception as e:
        status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
