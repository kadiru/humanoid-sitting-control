"""Matched physical and zero-child validation; does not train PPO."""
from pathlib import Path
import json,hashlib,shutil,traceback
import numpy as np
from run_transfer_assessment import run,write,status
from verify_stable_effort_leaning import starts,sha

ROOT=Path('/home/kadir/humanoid-sitting');DATA=Path('/data/humanoid-sitting-perception/contact_effort_repair_01');OLD=Path('/data/humanoid-sitting-perception/split_support_curriculum_02')
def read(p):return json.loads(p.read_text())
def command(folder,label,mode,seed,checkpoint=None,child=False,probe=False):
    cmd=['bash','scripts/run_supported_gpu0.sh','../scripts/'+('run_stable_effort_sitting.py' if child else 'run_contact_effort_repair.py'),'--bank',str(DATA/'banks/development'),'--output',str(folder/label),'--mode',mode,'--surface','sitting','--profiles',str(DATA/'sitting_repaired_profile.json'),'--seed',str(seed),'--start-stage','0','--num-envs','18','--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false','--plane-calibration',str(DATA/'plane_calibration.json')]
    if checkpoint:cmd+=['--parent-checkpoint' if child else '--checkpoint',checkpoint]
    if child:cmd+=['--calibration',str(folder/'calibration.json')]
    if probe:cmd+=['--reset-probe']
    return cmd

def worker():
    folder=DATA/'sitting_preflight_01';folder.mkdir(exist_ok=False)
    sources=[ROOT/'scripts/run_contact_effort_repair.py',ROOT/'scripts/run_stable_effort_sitting.py',ROOT/'sitting_task/src/humanoid_sitting/contact_aware_reference.py',Path(__file__).resolve(),DATA/'sitting_repaired_profile.json',DATA/'plane_calibration.json']
    (folder/'frozen').mkdir()
    for i,p in enumerate(sources):shutil.copy2(p,folder/'frozen'/f'{i:02d}_{p.name}')
    write(folder/'source_freeze.json',{str(p):sha(p) for p in sources})
    try:
        status(folder/'status.json','reference')
        run(folder,'reference',command(folder,'reference','reference',55501))
        ref=read(folder/'reference/results.json');assert ref['successes']==ref['count']==18,ref['successes']
        cal={}
        for row in ref['episodes']:
            g=read(folder/'reference/geometry.json')[row['geometry_index']];cal.setdefault(g['label'],[]).append(row['metrics_last10s']['total_squared_torque_Nm2'])
        cal={k:float(np.mean(v)) for k,v in cal.items()};cal['__pooled_development__']=float(np.mean(list(cal.values())));write(folder/'calibration.json',cal)
        run(folder,'reset_isolation',command(folder,'reset_isolation','reference',55501,probe=True))
        report=dict(passed=False,reference_successes=18,starts={},identity={})
        previous=Path('/data/humanoid-sitting-perception/transfer_effort_01/assessment/sitting')
        report['starts']['old_reference']=starts(previous/'reference',folder/'reference')
        for seed in [55211,55217]:
            cp=read(OLD/'ppo_01/sitting'/f'seed_{seed}.json')['last_checkpoint']
            label=f'parent_{seed}';status(folder/'status.json','parent_validation',active=label);run(folder,label,command(folder,label,'evaluate',55501,cp))
            result=read(folder/label/'results.json');assert result['successes']==result['count']==18,(label,result['successes'])
            report['starts'][label]=starts(previous/f'policy_{seed}',folder/label)
            zero=f'zero_{seed}';status(folder/'status.json','zero_child_validation',active=zero);run(folder,zero,command(folder,zero,'evaluate',55501,cp,child=True))
            for file in ['startup.npz','trace.npz']:
                a=np.load(folder/label/file);b=np.load(folder/zero/file)
                assert a.files==b.files,(file,a.files,b.files)
                assert all(np.array_equal(a[k],b[k]) for k in a.files),(label,file)
            assert read(folder/label/'results.json')==read(folder/zero/'results.json')
            report['identity'][str(seed)]=dict(startup_and_all_trace_arrays_exact=True,all_results_exact=True)
        for path,h in read(folder/'source_freeze.json').items():assert sha(path)==h,path
        prior=read(DATA/'prior_preservation.json')['hashes']
        for path,h in prior.items():assert sha(path)==h,path
        assert read(folder/'reset_isolation/reset_probe.json')['passed']
        report.update(passed=True,prior_hashes=len(prior),reset_isolation=True);write(folder/'verification.json',report);status(folder/'status.json','preflight_passed')
    except Exception as e:
        status(folder/'status.json','failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':worker()
