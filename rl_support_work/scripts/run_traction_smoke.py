"""Eight-rollout optimizer smoke followed by a full20s matched evaluation."""
import argparse,traceback
from run_traction_study import DATA,command
from run_transfer_assessment import run,status
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args();state=DATA/f'smoke_status_{a.seed}.json'
try:
    label=f'smoke_{a.seed}';status(state,'smoke_training')
    run(DATA,label,command(DATA,a.gpu,label,'train',a.seed,updates=8,warmup=2))
    checkpoint=DATA/label/'final.pt';label=f'smoke_eval_{a.seed}';status(state,'smoke_evaluation')
    run(DATA,label,command(DATA,a.gpu,label,'evaluate',checkpoint=checkpoint));status(state,'complete')
except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
