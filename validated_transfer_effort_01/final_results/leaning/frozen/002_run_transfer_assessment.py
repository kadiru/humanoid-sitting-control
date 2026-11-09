"""Independent persistent evaluation workers, not coding agents."""
import argparse,json,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/transfer_effort_01')
OLD=Path('/data/humanoid-sitting-perception/split_support_curriculum_02')
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2));tmp.replace(p)
def status(p,phase,**kw):write(p,dict(phase=phase,utc=datetime.now(timezone.utc).isoformat(),**kw))
def run(folder,label,cmd):
    assert not (folder/(label+'.log')).exists()
    write(folder/(label+'.command.json'),cmd)
    with (folder/(label+'.log')).open('w') as f:r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
    write(folder/(label+'.exit.json'),{'returncode':r.returncode});assert r.returncode==0,(label,r.returncode)
def command(folder,surface,gpu,label,mode,bank='development',checkpoint=None,count=18,script='run_transfer_effort.py',probe=False):
    profile=OLD/'split_profiles'/('sitting_selected.json' if surface=='sitting' else 'leaning_selected_02.json')
    cmd=['bash','scripts/'+('run_supported_gpu0.sh' if gpu==0 else 'run_resting_gpu1.sh'),'../scripts/'+script,'--bank',str(DATA/'banks'/bank),'--output',str(folder/label),'--mode',mode,'--surface',surface,'--profiles',str(profile),'--seed','55501','--start-stage','0','--num-envs',str(count),'--device','cuda:0',f'--kit_args=--/renderer/activeGpu={gpu},--/renderer/multiGpu/enabled=false']
    if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
    if probe:cmd+=['--reset-probe']
    if script!='run_transfer_effort.py':cmd[cmd.index('--bank')+1]='/data/humanoid-sitting-perception/two_surface_curriculum_01/selected_path_01'
    return cmd
def main():
    p=argparse.ArgumentParser();p.add_argument('--surface',choices=['sitting','leaning'],required=True);p.add_argument('--gpu',type=int,required=True);a=p.parse_args()
    folder=DATA/'assessment'/a.surface;folder.mkdir(parents=True,exist_ok=False)
    try:
        # Equal layout checks the generalized harness against the frozen controller.
        jobs=[('identity_frozen','reference','identity',None,2,'run_split_support.py' if a.surface=='sitting' else 'run_split_leaning.py',False),
              ('identity_new','reference','identity',None,2,'run_transfer_effort.py',False),
              ('reset_isolation','reference','development',None,18,'run_transfer_effort.py',True),
              ('fixed','fixed','development',None,18,'run_transfer_effort.py',False),
              ('reference','reference','development',None,18,'run_transfer_effort.py',False)]
        seeds=[55211,55217] if a.surface=='sitting' else [55311,55317]
        for seed in seeds:
            state=json.loads((OLD/'ppo_01'/a.surface/f'seed_{seed}.json').read_text())
            jobs.append((f'policy_{seed}','evaluate','development',state['last_checkpoint'],18,'run_transfer_effort.py',False))
        for label,mode,bank,checkpoint,count,script,probe in jobs:
            status(folder/'status.json','running',active=label)
            run(folder,label,command(folder,a.surface,a.gpu,label,mode,bank,checkpoint,count,script,probe))
            if label=='identity_new':
                import numpy as np
                for name in ['startup.npz','trace.npz']:
                    x=np.load(folder/'identity_frozen'/name);y=np.load(folder/'identity_new'/name)
                    assert x.files==y.files
                    for k in x.files:assert np.array_equal(x[k],y[k]),(name,k,float(abs(x[k]-y[k]).max()))
                write(folder/'identity_verification.json',{'passed':True,'startup_and_all_trace_arrays_exact':True})
        status(folder/'status.json','assessment_complete')
    except Exception as e:
        status(folder/'status.json','execution_failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
