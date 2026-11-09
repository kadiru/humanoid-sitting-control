"""Persistent physical probe worker, not a coding agent."""
import argparse,json,traceback
from pathlib import Path
from run_traction_study import command
from run_transfer_assessment import run,write,status
ROOT=Path('/data/humanoid-sitting-perception/posture_credit_01')
p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
jobs=[('hip_pitch',.008),('knee',.008),('ankle_pitch',.008)] if a.gpu==0 else [('hip_pitch',-.008),('knee',-.008),('ankle_pitch',-.008)]
state=ROOT/f'probe_status_gpu{a.gpu}.json'
try:
 for joint,offset in jobs:
  name=joint+('_positive' if offset>0 else '_negative')
  status(state,'running',active=name)
  cmd=command(ROOT/'probes',a.gpu,name,'reference');cmd[2]='../scripts/probe_posture_credit.py';cmd+=['--probe-joint',joint,'--probe-offset',str(offset)]
  run(ROOT/'probes',name,cmd)
  rows=json.loads((ROOT/'probes'/name/'results.json').read_text())['episodes']
  write(ROOT/'probes'/(name+'_summary.json'),dict(stable=sum(x['success'] for x in rows),strict=sum(x['strict_transfer'] for x in rows),physical=sum(any(x['failures'].values()) for x in rows),slip=sum(x['slip_ticks']>0 for x in rows),illegal=sum(x['illegal_ticks']>0 for x in rows),last5_torque=sum(x['metrics_last5s']['total_squared_torque_Nm2'] for x in rows)/len(rows)))
 status(state,'complete')
except Exception as e:
 status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
