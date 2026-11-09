"""Bounded independent leaning study, after its preflight passes.

This worker never trains sitting or changes prior research definitions.
"""
import argparse,json,shutil,sys,traceback
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from run_transfer_assessment import run,write,status
from verify_stable_effort_leaning import load,equal,weights,starts,sha,check_manifest,evaluation
from verify_split_leaning import evaluation as old_evaluation
from verify_transfer_effort_study import compare

ROOT=Path('/home/kadir/humanoid-sitting')
DATA=Path('/data/humanoid-sitting-perception/contact_effort_repair_01')
PREVIOUS=Path('/data/humanoid-sitting-perception/transfer_effort_01')
OLD=Path('/data/humanoid-sitting-perception/split_support_curriculum_02')
SEEDS=[56111,56117];PARENTS=[55311,55317]


def read(p):return json.loads(Path(p).read_text())
def parent(seed):return read(OLD/'ppo_01/leaning'/f'seed_{seed}.json')['last_checkpoint']


def command(folder,label,mode,seed,parent_checkpoint=None,checkpoint=None,bank='development',iterations=None,warmup=0,legacy=False):
    cmd=['bash','scripts/run_resting_gpu1.sh','../scripts/'+('run_transfer_effort.py' if legacy else 'run_stable_effort_leaning.py'),'--bank',str(DATA/'banks'/bank),'--output',str(folder/label),'--mode',mode,'--surface','leaning','--profiles',str(OLD/'split_profiles/leaning_selected_02.json'),'--seed',str(seed),'--start-stage','0','--num-envs','18' if bank=='development' else '16','--device','cuda:0','--kit_args=--/renderer/activeGpu=1,--/renderer/multiGpu/enabled=false']
    if parent_checkpoint:cmd+=['--parent-checkpoint',str(parent_checkpoint)]
    if checkpoint:cmd+=['--checkpoint',str(checkpoint)]
    if not legacy:cmd+=['--calibration',str(PREVIOUS/'calibration/leaning.json'),'--critic-warmup',str(warmup)]
    if iterations is not None:cmd+=['--iterations',str(iterations)]
    return cmd


def compare_pair(folder,label,baseline,parent_rows):
    rows=read(folder/label/'results.json')['episodes'];geoms=read(folder/label/'geometry.json')
    required={r['case'] for r,p in zip(baseline,parent_rows) if r['success'] or p['success']}
    successes={r['case'] for r in rows if r['success']};failures={r['case'] for r in rows if any(r['failures'].values())}
    comps={'corrected_reference':compare(rows,baseline,geoms),'parent':compare(rows,parent_rows,geoms)}
    improved=required<=successes and not failures
    for c in comps.values():
        for window in ('metrics_whole_trial','metrics_last5s'):
            metric=c['grouped'][window]['total_squared_torque_Nm2']
            improved &= metric['common_successes']==len(required) and metric['change_percent']<0
    return dict(stable=required<=successes and not failures,eligible=bool(improved),successes=len(successes),physical_failures=len(failures),lost=sorted(required-successes),comparisons=comps)


def verify(folder,output):
    output.mkdir(exist_ok=False);p=read(folder/'protocol.json')
    for path,h in p['hashes'].items():assert sha(path)==h,path
    for path,h in p['prior_hashes'].items():assert sha(path)==h,path
    report=dict(passed=False,frozen_hashes=len(p['hashes']),prior_hashes=len(p['prior_hashes']),training={},evaluations={},starts={},comparisons={},selection={})
    ref=read(PREVIOUS/'assessment/leaning/reference/results.json')['episodes']
    for seed,oldseed in zip(SEEDS,PARENTS):
        state=read(folder/f'seed_{seed}.json');previous=None;budget=dict(updates=0,actor_updates=0,policy_decisions=0,transitions=0,active=0,warmup=0)
        for block,expected in ((0,24),(1,16)):
            f=folder/f'train_{seed}_block{block}';m=check_manifest(f);v=read(f/'verification.json');initial=load(f/'initial.pt');final=load(f/'final.pt')
            assert v['updates']==expected and v['actor_updates']==16 and v['policy_decisions']==expected*16*18 and v['transitions']==expected*16*18*10
            assert v['active']+v['warmup']==v['transitions']
            weights(initial,final)
            assert initial['actor_state_dict']['mlp.0.weight'].shape[1]==406 and initial['actor_state_dict']['mlp.4.weight'].shape[0]==2
            assert m['args']['parent_checkpoint']==parent(oldseed) and read(f/'parent_identity.json')['sha256']==sha(parent(oldseed))
            if previous:
                prev=load(previous);assert all(equal(initial[k],prev[k]) for k in ('actor_state_dict','critic_state_dict','optimizer_state_dict'))
                assert initial['iter']==prev['iter']+1
            else:
                warm=load(f/'critic_warmup.pt');assert equal(initial['actor_state_dict'],warm['actor_state_dict'])
                assert initial['actor_state_dict']['mlp.4.weight'].count_nonzero()==0 and initial['actor_state_dict']['mlp.4.bias'].count_nonzero()==0
            assert final['iter']==initial['iter']+expected-1
            for k in budget:budget[k]+=v[k]
            rows=read(f/'results.json')['episodes'];pending=25*len(rows)-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in rows)
            assert 0<=pending<=450 and 0<=unfinished<=18000
            previous=f/'final.pt'
            label=f'dev_{seed}_block{block}';ev,_=evaluation(folder/label,18,previous);report['evaluations'][label]=ev
            report['starts'][label]=starts(PREVIOUS/'assessment/leaning/reference',folder/label)
            check=compare_pair(folder,label,ref,read(PREVIOUS/'assessment/leaning'/f'policy_{oldseed}/results.json')['episodes'])
            assert check==state['history'][block]['gate'];report['comparisons'][label]=check['comparisons']
        assert budget['transitions']==115200 and budget['actor_updates']==32
        eligible=[x for x in state['history'] if x['gate']['eligible']]
        best=min(eligible,key=lambda x:x['gate']['comparisons']['corrected_reference']['grouped']['metrics_whole_trial']['total_squared_torque_Nm2']['policy']) if eligible else None
        assert state['selected_checkpoint']==(best['checkpoint'] if best else None)
        report['selection'][str(seed)]=dict(selected=state['selected_checkpoint'],parent_fallback=best is None,used_holdout=False)
        report['training'][str(seed)]=budget
    frozen=read(folder/'selection_frozen.json')
    jobs=[('holdout_reference',None,True)]
    for seed,oldseed in zip(SEEDS,PARENTS):
        state=read(folder/f'seed_{seed}.json')
        jobs.extend([(f'holdout_parent_{oldseed}',parent(oldseed),True),(f'holdout_final_{seed}',state['final_checkpoint'],False)])
        if state['selected_checkpoint'] and state['selected_checkpoint']!=state['final_checkpoint']:jobs.append((f'holdout_selected_{seed}',state['selected_checkpoint'],False))
    for label,cp,legacy in jobs:
        ev,_=(old_evaluation if legacy else evaluation)(folder/label,16,Path(cp) if cp else None)
        report['evaluations'][label]=ev;report['starts'][label]=starts(folder/'holdout_reference',folder/label)
        assert (folder/label/'manifest.json').stat().st_mtime>=frozen['unix_time']
        if not legacy:
            seed=int(label.rsplit('_',1)[1]);oldseed=PARENTS[SEEDS.index(seed)]
            report['comparisons'][label]=compare_pair(folder,label,read(folder/'holdout_reference/results.json')['episodes'],read(folder/f'holdout_parent_{oldseed}/results.json')['episodes'])['comparisons']
    report['passed']=True;write(output/'verification.json',report);shutil.copy2(__file__,output/'verification_source.py')
    return report


def worker():
    assert read(DATA/'leaning_preflight_01/verification.json')['passed']
    folder=DATA/'leaning_ppo_01';folder.mkdir(exist_ok=False);(folder/'frozen').mkdir()
    paths=[ROOT/'scripts'/n for n in ('run_stable_effort_leaning.py','run_stable_effort_comparison.py','verify_stable_effort_leaning.py','run_resting_posture.py','run_supported_postures.py','run_supported_transition.py','run_resting_gpu1.sh')]
    paths += [Path(x) for x in read(DATA/'leaning_preflight_01/zero/manifest.json')['sources']]
    paths += [x for x in (DATA/'banks').rglob('*') if x.is_file()]
    paths += [DATA/'leaning_preflight_01/verification.json',ROOT/'docs/CONTACT_EFFORT_REPAIR_PROTOCOL.md',PREVIOUS/'calibration/leaning.json',OLD/'split_profiles/leaning_selected_02.json']
    paths += [Path(parent(s)) for s in PARENTS]
    hashes={str(x):sha(x) for x in paths};prior=read(DATA/'prior_preservation.json')['hashes']
    for path,h in prior.items():assert sha(path)==h,path
    for i,path in enumerate(hashes):shutil.copy2(path,folder/'frozen'/f'{i:03d}_{Path(path).name}')
    write(folder/'protocol.json',dict(task='leaning',seeds=SEEDS,parent_seeds=PARENTS,blocks=[dict(updates=24,critic_only=8,actor_updates=16),dict(updates=16,critic_only=0,actor_updates=16)],policy_steps_per_update=16,physics_ticks_per_action=10,num_envs=18,physics_transitions_per_seed=115200,total_physics_transitions=230400,actor_observations=406,parent_observations=405,child_actions=2,hashes=hashes,prior_hashes=prior,selection='Must retain all reference/parent successes, no physical failures, and improve whole20s AND final5s squared torque against BOTH baselines. Otherwise parent fallback. Development only.'))
    try:
        ref=read(PREVIOUS/'assessment/leaning/reference/results.json')['episodes']
        for seed,oldseed in zip(SEEDS,PARENTS):
            cp=None;history=[];par=parent(oldseed);rows=read(PREVIOUS/'assessment/leaning'/f'policy_{oldseed}/results.json')['episodes']
            for block in range(2):
                label=f'train_{seed}_block{block}';status(folder/'status.json','training',active=label)
                run(folder,label,command(folder,label,'train',seed,par,cp,iterations=24 if block==0 else 16,warmup=8 if block==0 else 0));cp=str(folder/label/'final.pt')
                label=f'dev_{seed}_block{block}';status(folder/'status.json','development_evaluation',active=label)
                run(folder,label,command(folder,label,'evaluate',55501,par,cp))
                gate=compare_pair(folder,label,ref,rows);history.append(dict(block=block,checkpoint=cp,gate=gate))
                write(folder/f'seed_{seed}.json',dict(parent=par,history=history))
            eligible=[h for h in history if h['gate']['eligible']]
            best=min(eligible,key=lambda x:x['gate']['comparisons']['corrected_reference']['grouped']['metrics_whole_trial']['total_squared_torque_Nm2']['policy']) if eligible else None
            write(folder/f'seed_{seed}.json',dict(parent=par,history=history,final_checkpoint=cp,selected_checkpoint=best['checkpoint'] if best else None,parent_fallback=best is None))
        write(folder/'selection_frozen.json',dict(unix_time=datetime.now(timezone.utc).timestamp(),seeds=SEEDS,selection={str(s):read(folder/f'seed_{s}.json')['selected_checkpoint'] for s in SEEDS},scope='Before any new holdout evaluation'))
        jobs=[('holdout_reference','reference',None,None,True)]
        for seed,oldseed in zip(SEEDS,PARENTS):
            s=read(folder/f'seed_{seed}.json');par=parent(oldseed)
            jobs.extend([(f'holdout_parent_{oldseed}','evaluate',None,par,True),(f'holdout_final_{seed}','evaluate',par,s['final_checkpoint'],False)])
            if s['selected_checkpoint'] and s['selected_checkpoint']!=s['final_checkpoint']:jobs.append((f'holdout_selected_{seed}','evaluate',par,s['selected_checkpoint'],False))
        for label,mode,par,cp,legacy in jobs:
            status(folder/'status.json','holdout_evaluation',active=label)
            run(folder,label,command(folder,label,mode,55501,par,cp,bank='holdout',legacy=legacy))
        status(folder/'status.json','verifying');verify(folder,folder/'verified_01');status(folder/'status.json','verified_complete')
    except Exception as e:
        status(folder/'status.json','failed',error=repr(e),traceback=traceback.format_exc());raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--verify',type=Path);parser.add_argument('--output',type=Path);args=parser.parse_args()
    if args.verify:verify(args.verify,args.output)
    else:worker()
