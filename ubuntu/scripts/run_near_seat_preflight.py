"""Execution preflight only; no main-study learning."""
import argparse,traceback
from run_progressive_seat_study import DATA,command,run,status,read,write

def main():
 p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();folder=DATA/'preflight_01';folder.mkdir(exist_ok=True)
 jobs=[('learned_smoke','evaluate',5,DATA/'optimizer_smoke_01/final.pt',False),('reference_stage4','reference',4,None,False)] if a.gpu==0 else [(f'reset_stage{s}','reference',s,None,True) for s in [1,2,3,4,5]]
 try:
  for label,mode,stage,cp,probe in jobs:
   status(folder/f'status_gpu{a.gpu}.json','running',active=label)
   run(folder,label,command(folder,a.gpu,label,mode,stage,checkpoint=cp,probe=probe))
   initial=read(folder/label/'startup_validation.json');assert not any(any(v) for v in initial['failures'].values());assert all(initial['sensor_valid'])
   if probe:assert read(folder/label/'reset_probe.json')['passed']
  status(folder/f'status_gpu{a.gpu}.json','complete')
 except Exception as e:status(folder/f'status_gpu{a.gpu}.json','failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
