"""Persistent simulation worker for one frozen seat-load seed; not an agent."""
import argparse,hashlib,json,time,traceback
from pathlib import Path
from run_transfer_assessment import run,write,status

ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/seat_load_transfer_01')
OLD=Path('/data/humanoid-sitting-perception/contact_effort_repair_01')

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def check(p):
    for k,h in p['hashes'].items():assert sha(k)==h,k
    for k,h in p['prior_hashes'].items():assert sha(k)==h,k

def command(folder,gpu,label,mode,bank='development',seed=55501,kind='new',checkpoint=None):
    script={'new':'run_seat_load_ppo.py','current_reference':'run_contact_effort_repair.py','current_policy':'run_stable_effort_sitting.py'}[kind]
    profile=DATA/'selected_profile.json' if kind=='new' else OLD/'sitting_repaired_profile.json'
    cmd=['bash','scripts/'+('run_supported_gpu0.sh' if gpu==0 else 'run_resting_gpu1.sh'),'../scripts/'+script,'--bank',str(DATA/'banks'/bank),'--output',str(folder/label),'--mode',mode,'--surface','sitting','--profiles',str(profile),'--plane-calibration',str(OLD/'plane_calibration.json'),'--seed',str(seed),'--start-stage','0','--num-envs','18' if bank=='development' else '16','--device','cuda:0',f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']
    if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
    if mode=='train':cmd+=['--iterations','64','--critic-warmup','8']
    if kind=='current_policy':
        parent=read(Path('/data/humanoid-sitting-perception/split_support_curriculum_02/ppo_01/sitting/seed_55211.json'))['last_checkpoint']
        cmd+=['--parent-checkpoint',parent,'--calibration',str(OLD/'sitting_preflight_01/calibration.json')]
    return cmd

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,choices=[0,1],required=True);p.add_argument('--seed',type=int,choices=[56811,56817],required=True);a=p.parse_args()
    folder=DATA/'ppo_01';protocol=read(folder/'protocol.json');check(protocol);state=folder/f'status_{a.seed}.json'
    try:
        label=f'train_{a.seed}';status(state,'training',active=label);run(folder,label,command(folder,a.gpu,label,'train',seed=a.seed))
        cp=folder/label/'final.pt';assert read(folder/label/'verification.json')['updates']==64
        write(folder/f'training_finished_{a.seed}.json',dict(checkpoint=str(cp),sha256=sha(cp),time=time.time()))
        label=f'development_{a.seed}';status(state,'development_evaluation',active=label);run(folder,label,command(folder,a.gpu,label,'evaluate',checkpoint=cp))
        status(state,'waiting_for_both_final_checkpoints')
        while not all((folder/f'training_finished_{seed}.json').exists() for seed in [56811,56817]):time.sleep(10)
        barrier=folder/'holdout_barrier.json'
        if a.gpu==0:write(barrier,dict(time=time.time(),selection='Full-budget final only; no development or holdout checkpoint selection',checkpoints={str(s):read(folder/f'training_finished_{s}.json') for s in [56811,56817]}))
        while not barrier.exists():time.sleep(5)
        jobs=[]
        if a.gpu==0:jobs=[('holdout_current_reference','reference','current_reference',None),('holdout_new_reference','reference','new',None)]
        else:
            jobs=[(f'holdout_current_{s}','evaluate','current_policy',OLD/f'sitting_ppo_01/train_{s}/final.pt') for s in [56211,56217]]
        jobs.append((f'holdout_{a.seed}','evaluate','new',cp))
        for label,mode,kind,checkpoint in jobs:
            status(state,'holdout_evaluation',active=label);run(folder,label,command(folder,a.gpu,label,mode,bank='holdout',kind=kind,checkpoint=checkpoint))
        check(protocol);status(state,'complete',checkpoint=str(cp))
    except Exception as e:
        status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
