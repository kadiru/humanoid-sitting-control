"""Persistent full-budget worker with final-checkpoint holdout barrier."""
import argparse,time,traceback
from posture_credit_common import *
from run_transfer_assessment import run,write,status
from run_traction_study import command as traction_command
STUDY=DATA/'ppo_01'

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();seed=SEEDS[a.gpu]
 protocol=read(STUDY/'protocol.json');assert protocol['admitted'];check_hashes(protocol['hashes']);check_hashes(protocol['prior_hashes'])
 folder=STUDY/f'seed_{seed}';folder.mkdir(exist_ok=False);state=STUDY/f'status_{seed}.json'
 try:
  status(state,'training',seed=seed,updates=10)
  run(folder,'train',command(folder,a.gpu,'train','train',seed=seed,updates=10))
  budget=read(folder/'train/episode_contract.json');assert budget['updates']==10 and budget['transitions']==184050 and budget['episodes']==180
  cp=folder/'train/final.pt';final=dict(checkpoint=str(cp),sha256=sha(cp),timestamp=time.time(),updates=10)
  write(STUDY/f'training_finished_{seed}.json',final)
  status(state,'development_evaluation',**final)
  run(folder,'development_final',command(folder,a.gpu,'development_final','evaluate',checkpoint=cp))
  status(state,'waiting_for_both_final_checkpoints',**final)
  while not all((STUDY/f'training_finished_{s}.json').exists() for s in SEEDS):
   for s in SEEDS:
    f=STUDY/f'status_{s}.json';assert not f.exists() or read(f)['phase']!='failed',f
   time.sleep(15)
  barrier=STUDY/'holdout_barrier.json'
  if a.gpu==0:write(barrier,dict(timestamp=time.time(),finals={str(s):read(STUDY/f'training_finished_{s}.json') for s in SEEDS},selection='Full-budget final only; no holdout selection'))
  while not barrier.exists():time.sleep(5)
  if a.gpu==0:
   label='holdout_reference';status(state,'holdout_evaluation',active=label)
   run(STUDY,label,command(STUDY,0,label,'zero',holdout=True))
  oldseed=[58311,58317][a.gpu];label=f'holdout_previous_{oldseed}'
  status(state,'holdout_evaluation',active=label)
  cmd=traction_command(STUDY,a.gpu,label,'evaluate',checkpoint=OLD/f'ppo_01/seed_{oldseed}/train/final.pt',holdout=True)
  cmd[cmd.index('--bank')+1]=str(DATA/'reserved_bank_01');run(STUDY,label,cmd)
  label=f'holdout_policy_{seed}';status(state,'holdout_evaluation',active=label)
  run(STUDY,label,command(STUDY,a.gpu,label,'evaluate',checkpoint=cp,holdout=True))
  check_hashes(protocol['hashes']);check_hashes(protocol['prior_hashes']);status(state,'complete',**final)
 except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
