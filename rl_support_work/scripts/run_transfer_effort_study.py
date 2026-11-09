"""Persistent independent effort-training workers with frozen development-only selection."""
import argparse,hashlib,json,shutil,subprocess,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from run_transfer_assessment import DATA,OLD,ROOT,write,status,run,command
from verify_split_leaning import evaluation,starts,read,sha,load,equal,weights

def seeds(surface):return [55611,55617] if surface=='sitting' else [55711,55717]
def parents(surface):return [55211,55217] if surface=='sitting' else [55311,55317]
def parent_path(surface,seed):return Path(read(OLD/'ppo_01'/surface/f'seed_{seed}.json')['last_checkpoint'])
def cmd(folder,surface,gpu,label,mode,seed,checkpoint=None,bank='development',iterations=32):
    out=command(folder,surface,gpu,label,mode,bank,checkpoint,18 if bank=='development' else 16)
    out[out.index('--seed')+1]=str(seed)
    if mode=='train':out+=['--iterations',str(iterations),'--objective','effort','--calibration',str(DATA/'calibration'/f'{surface}.json')]
    return out

def retention(rows,ref,parent):
    assert len(rows)==len(ref)==len(parent)
    required=[i for i,(a,b) in enumerate(zip(ref,parent)) if a['success'] or b['success']]
    previous_fail={i for i,(a,b) in enumerate(zip(ref,parent)) if any(a['failures'].values()) or any(b['failures'].values())}
    now_fail={i for i,x in enumerate(rows) if any(x['failures'].values())}
    lost=[i for i in required if not rows[i]['success']]
    eligible=not lost and not (now_fail-previous_fail)
    common=[x['metrics_whole_trial']['total_squared_torque_Nm2'] for x,b in zip(rows,ref) if x['success'] and b['success']]
    return dict(eligible=eligible,lost_success_conditions=lost,new_physical_failures=sorted(now_fail-previous_fail),successes=sum(x['success'] for x in rows),reference_common_count=len(common),mean_squared_torque=float(np.mean(common)) if common else None)

def make_calibration(surface):
    assess=DATA/'assessment'/surface;v=read(assess/'verified_01/verification.json');assert v['passed']
    rows=read(assess/'reference/results.json')['episodes'];geoms=read(assess/'reference/geometry.json')
    good=[x['metrics_last10s']['total_squared_torque_Nm2'] for x in rows if x['success']];assert good,'No successful corrected-reference trial to calibrate'
    fallback=float(np.mean(good));cal={};details={}
    for i,g in enumerate(geoms):
        values=[x['metrics_last10s']['total_squared_torque_Nm2'] for x in rows if x['geometry_index']==i and x['success']]
        cal[g['label']]=float(np.mean(values)) if values else fallback
        details[g['label']]=dict(successful_reference_trials=len(values),reference_Nm2=cal[g['label']],source='matched successful full-duration trials' if values else 'pooled successful development reference; reward scaling only')
    p=DATA/'calibration'/f'{surface}.json';assert not p.exists();write(p,cal);write(p.with_name(surface+'_provenance.json'),details)

def verify_training(folder,parent,expected_updates):
    m=read(folder/'manifest.json');v=read(folder/'verification.json');initial=load(folder/'initial.pt');final=load(folder/'final.pt');prev=load(parent)
    assert all(equal(initial[k],prev[k]) for k in ['actor_state_dict','critic_state_dict','optimizer_state_dict'])
    assert initial['iter']==prev['iter']+1;assert final['iter']==initial['iter']+expected_updates-1
    assert v['updates']==expected_updates and v['transitions']==expected_updates*18*64 and v['active']+v['warmup']==v['transitions']
    changed=weights(initial,final)
    episodes=read(folder/'results.json')['episodes'];pending=len(episodes)*25-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in episodes)
    assert 0<=pending<=18*25 and 0<=unfinished<=18*1000
    assert m['args']['objective']=='effort' and m['args']['start_stage']=='0'
    return dict(budget=v,changed=changed,iteration=int(final['iter']),pending_warmup=pending,unfinished_active=unfinished,checkpoint_sha256=sha(folder/'final.pt'))

def preflight(surface,gpu):
    make_calibration(surface);folder=DATA/'preflight'/surface;folder.mkdir(parents=True,exist_ok=False)
    parent=parent_path(surface,parents(surface)[0]);seed=seeds(surface)[0];assess=DATA/'assessment'/surface
    try:
        status(folder/'status.json','optimizer_smoke')
        run(folder,'smoke_train',cmd(folder,surface,gpu,'smoke_train','train',seed,parent,iterations=2))
        check=verify_training(folder/'smoke_train',parent,2)
        run(folder,'smoke_evaluate',cmd(folder,surface,gpu,'smoke_evaluate','evaluate',55501,folder/'smoke_train/final.pt'))
        ev,results=evaluation(folder/'smoke_evaluate',18,folder/'smoke_train/final.pt')
        startup=starts(assess/'reference',folder/'smoke_evaluate')
        gate=retention(results['episodes'],read(assess/'reference/results.json')['episodes'],read(assess/f'policy_{parents(surface)[0]}/results.json')['episodes'])
        write(folder/'verification.json',dict(passed=gate['eligible'],training=check,evaluation=ev,startups=startup,retention=gate))
        assert gate['eligible'],('Smoke loses baseline stability',gate)
        status(folder/'status.json','preflight_passed')
    except Exception as e:status(folder/'status.json','execution_or_gate_failed',error=repr(e),traceback=traceback.format_exc());raise

def freeze(folder,surface,gpu):
    folder.mkdir(parents=True,exist_ok=False);(folder/'frozen').mkdir()
    assert read(DATA/'preflight'/surface/'verification.json')['passed']
    paths=[ROOT/'scripts'/n for n in ['run_transfer_effort.py','run_transfer_effort_study.py','run_transfer_assessment.py','verify_transfer_assessment.py','verify_transfer_effort_study.py','verify_split_leaning.py','test_transfer_effort.py','run_supported_transition.py','run_supported_postures.py','run_resting_posture.py','run_supported_gpu0.sh','run_resting_gpu1.sh']]
    paths+=[ROOT/'docs/TRANSFER_EFFORT_PROTOCOL.md',DATA/'banks/bank_protocol.json',DATA/'calibration'/f'{surface}.json',DATA/'preflight'/surface/'verification.json']
    for split in ['development','holdout']:paths+=list((DATA/'banks'/split).iterdir())
    paths+=list((ROOT/'sitting_task/src/humanoid_sitting/tasks').glob('*.py'))
    paths+=[ROOT/'sitting_task/src/humanoid_sitting'/n for n in ['two_surface_path.py','supported_surface.py','resting_objective.py']]
    paths+=[OLD/'split_profiles'/('sitting_selected.json' if surface=='sitting' else 'leaning_selected_02.json')]
    paths+=[parent_path(surface,s) for s in parents(surface)]
    paths+=[p for p in (DATA/'assessment'/surface).rglob('*') if p.is_file() and p.name in ['results.json','manifest.json','geometry.json','startup.npz','trace.npz','verification.json']]
    hashes={str(p):sha(p) for p in paths if p.is_file()};prior=read(DATA/'prior_preservation.json')['hashes']
    for p,h in prior.items():assert sha(p)==h,p
    for i,p in enumerate(hashes):shutil.copy2(p,folder/'frozen'/f'{i:03d}_{Path(p).name}')
    write(folder/'protocol.json',dict(surface=surface,gpu=gpu,seeds=seeds(surface),parent_seeds=parents(surface),blocks=4,updates_per_block=32,updates_per_seed=128,transitions_per_seed=147456,total_transitions=294912,stages=[0]*4,num_envs=18,rollout_steps=64,hashes=hashes,prior_hashes=prior,holdout_never_used_for_selection=True))

def worker(surface,gpu):
    folder=DATA/'ppo'/surface;freeze(folder,surface,gpu);assess=DATA/'assessment'/surface;ref=read(assess/'reference/results.json')['episodes']
    try:
        for seed,oldseed in zip(seeds(surface),parents(surface)):
            checkpoint=parent_path(surface,oldseed);parentrows=read(assess/f'policy_{oldseed}/results.json')['episodes'];history=[];eligible=[]
            for block in range(4):
                label=f'train_{seed}_block{block}';status(folder/'status.json','training',active=label,seed=seed)
                run(folder,label,cmd(folder,surface,gpu,label,'train',seed,checkpoint));budget=verify_training(folder/label,checkpoint,32);checkpoint=folder/label/'final.pt'
                label=f'dev_{seed}_block{block}';status(folder/'status.json','development_evaluation',active=label,seed=seed)
                run(folder,label,cmd(folder,surface,gpu,label,'evaluate',55501,checkpoint))
                ev,results=evaluation(folder/label,18,checkpoint);startup=starts(assess/'reference',folder/label);gate=retention(results['episodes'],ref,parentrows)
                row=dict(block=block,checkpoint=str(checkpoint),train=f'train_{seed}_block{block}',evaluation=label,verification=budget,retention=gate)
                history.append(row)
                if gate['eligible']:eligible.append(row)
                write(folder/f'seed_{seed}.json',dict(phase='training',parent=str(parent_path(surface,oldseed)),history=history))
            chosen=min(eligible,key=lambda x:(-x['retention']['successes'],x['retention']['mean_squared_torque'])) if eligible else None
            selected=chosen['checkpoint'] if chosen else str(parent_path(surface,oldseed))
            write(folder/f'seed_{seed}.json',dict(phase='trained_selection_frozen',parent=str(parent_path(surface,oldseed)),history=history,final_checkpoint=str(checkpoint),selected_checkpoint=selected,selected_block=chosen['block'] if chosen else None,selected_is_fallback=chosen is None,selection_scope='development only; no holdout yet',updates=128,transitions=147456))
        # Both seeds are fully trained and selected before any held-out result is computed.
        jobs=[('holdout_reference','reference',None)]
        for seed,oldseed in zip(seeds(surface),parents(surface)):
            state=read(folder/f'seed_{seed}.json')
            jobs.extend([(f'holdout_parent_{oldseed}','evaluate',str(parent_path(surface,oldseed))), (f'holdout_final_{seed}','evaluate',state['final_checkpoint'])])
            if state['selected_checkpoint'] not in [state['final_checkpoint'],state['parent']]:jobs.append((f'holdout_selected_{seed}','evaluate',state['selected_checkpoint']))
        for label,mode,checkpoint in jobs:
            status(folder/'status.json','holdout_evaluation',active=label)
            run(folder,label,cmd(folder,surface,gpu,label,mode,55501,checkpoint,bank='holdout'))
        status(folder/'status.json','complete_pending_verification')
        run(folder,'verification_01',[sys.executable,str(ROOT/'scripts/verify_transfer_effort_study.py'),'--data',str(folder),'--output',str(folder/'verified_01')])
        status(folder/'status.json','verified_complete')
    except Exception as e:status(folder/'status.json','execution_failed',error=repr(e),traceback=traceback.format_exc());raise

def main():
    p=argparse.ArgumentParser();p.add_argument('--surface',choices=['sitting','leaning'],required=True);p.add_argument('--gpu',type=int,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args()
    (preflight if a.preflight else worker)(a.surface,a.gpu)
if __name__=='__main__':main()
