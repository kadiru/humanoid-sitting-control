"""Persistent physical validation only; no PPO training in this driver."""
import argparse,json,traceback
from pathlib import Path
from run_progressive_seat_study import command,run,status,read

DATA=Path('/data/humanoid-sitting-perception/crouch_rest_bridge_01')
def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    jobs=[('continuous_01',[])] if a.gpu==0 else [('hold_01',['--lock-transfer']),('delayed_bias_01',['--bias-delay','.35'])]
    try:
        for label,extra in jobs:
            status(DATA/f'diagnostics_gpu{a.gpu}.json','running',active=label)
            cmd=command(DATA,a.gpu,label,'reference',stage=3)
            cmd[2]='../scripts/run_crouch_rest_probe.py';cmd+=extra
            run(DATA,label,cmd)
            result=read(DATA/label/'results.json');print(label,result['successes'],sum(r['strict_transfer'] for r in result['episodes']),flush=True)
        status(DATA/f'diagnostics_gpu{a.gpu}.json','complete')
    except Exception as e:
        status(DATA/f'diagnostics_gpu{a.gpu}.json','failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
