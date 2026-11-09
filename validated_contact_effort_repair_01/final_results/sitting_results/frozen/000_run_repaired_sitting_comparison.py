"""Frozen bounded sitting comparison; validates admission before any training."""
import json,shutil,traceback,argparse
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from run_transfer_assessment import run,write,status
from verify_repaired_sitting import check_manifest,sha,load,equal,weights,starts,evaluation,plan,approach
from verify_split_leaning import evaluation as old_evaluation
from verify_transfer_effort_study import compare

ROOT=Path('/home/kadir/humanoid-sitting');DATA=Path('/data/humanoid-sitting-perception/contact_effort_repair_01');PRE=DATA/'sitting_preflight_01';OLD=Path('/data/humanoid-sitting-perception/split_support_curriculum_02');PREVIOUS=Path('/data/humanoid-sitting-perception/transfer_effort_01/assessment/sitting')
SEEDS=[56211,56217];PARENTS=[55211,55211];BASELINES=[55211,55217]
def read(p):return json.loads(p.read_text())
def parent(s):return read(OLD/'ppo_01/sitting'/f'seed_{s}.json')['last_checkpoint']
def command(folder,label,mode,seed,bank='development',kind='child',parent_cp=None,cp=None):
    script={'child':'run_stable_effort_sitting.py','repaired':'run_contact_effort_repair.py','old':'run_transfer_effort.py'}[kind]
    profile=OLD/'split_profiles/sitting_selected.json' if kind=='old' else DATA/'sitting_repaired_profile.json'
    cmd=['bash','scripts/run_supported_gpu0.sh','../scripts/'+script,'--bank',str(DATA/'banks'/bank),'--output',str(folder/label),'--mode',mode,'--surface','sitting','--profiles',str(profile),'--seed',str(seed),'--start-stage','0','--num-envs','18' if bank=='development' else '16','--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false']
    if kind!='old':cmd+=['--plane-calibration',str(DATA/'plane_calibration.json')]
    if kind=='child':cmd+=['--calibration',str(PRE/'calibration.json'),'--parent-checkpoint',str(parent_cp),'--critic-warmup','8']
    if cp:cmd+=['--checkpoint',str(cp)]
    if mode=='train':assert kind=='child';cmd+=['--iterations','40']
    return cmd

def comparison(folder,label,refdir,parentdir):
    rows=read(folder/label/'results.json')['episodes'];reference=read(refdir/'results.json')['episodes'];parents=read(parentdir/'results.json')['episodes'];geoms=read(folder/label/'geometry.json')
    required={a['case'] for a,b in zip(reference,parents) if a['success'] or b['success']};success={x['case'] for x in rows if x['success']};fail={x['case'] for x in rows if any(x['failures'].values())}
    comps={'repaired_reference':compare(rows,reference,geoms),'repaired_starting_policy':compare(rows,parents,geoms)}
    eligible=required<=success and not fail
    for c in comps.values():
        for window in ['metrics_whole_trial','metrics_last5s']:
            m=c['grouped'][window]['total_squared_torque_Nm2'];eligible &= m['common_successes']==len(required) and m['change_percent']<0
    return dict(eligible=bool(eligible),successes=len(success),physical_failures=len(fail),lost=sorted(required-success),comparisons=comps)

def verify(folder,out):
    out.mkdir(exist_ok=False);p=read(folder/'protocol.json')
    for path,h in p['hashes'].items():assert sha(path)==h,path
    for path,h in p['prior_hashes'].items():assert sha(path)==h,path
    report=dict(passed=False,frozen_hashes=len(p['hashes']),prior_hashes=len(p['prior_hashes']),training={},evaluations={},starts={},plans={},approach_identity={},comparisons={},selection={})
    for seed,oldseed in zip(SEEDS,PARENTS):
        state=read(folder/f'seed_{seed}.json');f=folder/f'train_{seed}';v=read(f/'verification.json');m=check_manifest(f);initial=load(f/'initial.pt');warm=load(f/'critic_warmup.pt');final=load(f/'final.pt')
        assert (v['updates'],v['actor_updates'],v['policy_decisions'],v['transitions'])==(40,32,11520,115200)
        assert v['active']+v['warmup']==v['transitions'];weights(initial,final);assert equal(initial['actor_state_dict'],warm['actor_state_dict'])
        assert initial['actor_state_dict']['mlp.0.weight'].shape[1]==406 and initial['actor_state_dict']['mlp.4.weight'].shape[0]==2
        assert initial['actor_state_dict']['mlp.4.weight'].count_nonzero()==0 and initial['actor_state_dict']['mlp.4.bias'].count_nonzero()==0
        assert final['iter']==initial['iter']+39 and m['args']['parent_checkpoint']==parent(oldseed)
        assert read(f/'parent_identity.json')['sha256']==sha(parent(oldseed))
        rows=read(f/'results.json')['episodes'];pending=25*len(rows)-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in rows);assert 0<=pending<=450 and 0<=unfinished<=18000
        report['training'][str(seed)]=dict(**v,pending_warmup=pending,unfinished_active=unfinished)
        label=f'dev_{seed}';ev,_=evaluation(folder/label,18,f/'final.pt');report['evaluations'][label]=ev;report['plans'][label]=plan(folder/label)
        report['starts'][label]=starts(PRE/'reference',folder/label);report['approach_identity'][label]=approach(PRE/f'parent_{oldseed}',folder/label)
        gate=comparison(folder,label,PRE/'reference',PRE/f'parent_{oldseed}');assert gate==state['gate'];report['comparisons'][label]=gate['comparisons']
        assert state['parent_fallback']==(not gate['eligible']);report['selection'][str(seed)]=dict(parent_fallback=state['parent_fallback'],used_holdout=False)
    frozen=read(folder/'selection_frozen.json')
    jobs=[('holdout_old_reference',None,'old'),('holdout_repaired_reference',None,'repaired')]
    for oldseed in BASELINES:jobs += [(f'holdout_old_parent_{oldseed}',parent(oldseed),'old'),(f'holdout_repaired_parent_{oldseed}',parent(oldseed),'repaired')]
    for seed in SEEDS:jobs += [(f'holdout_final_{seed}',str(folder/f'train_{seed}/final.pt'),'child')]
    for label,cp,kind in jobs:
        ev,_=(old_evaluation if kind=='old' else evaluation)(folder/label,16,Path(cp) if cp else None);report['evaluations'][label]=ev
        report['starts'][label]=starts(folder/'holdout_old_reference',folder/label)
        assert (folder/label/'manifest.json').stat().st_mtime>=frozen['unix_time']
        if kind!='old':report['plans'][label]=plan(folder/label)
    for seed,oldseed in zip(SEEDS,PARENTS):
        label=f'holdout_final_{seed}';report['approach_identity'][label]=approach(folder/f'holdout_repaired_parent_{oldseed}',folder/label)
        rows=read(folder/label/'results.json')['episodes'];g=read(folder/label/'geometry.json')
        report['comparisons'][label]={name:compare(rows,read(folder/path/'results.json')['episodes'],g) for name,path in [('old_corrected_reference','holdout_old_reference'),('repaired_reference','holdout_repaired_reference')]+[(f'old_starting_policy_{s}',f'holdout_old_parent_{s}') for s in BASELINES]+[(f'repaired_starting_policy_{s}',f'holdout_repaired_parent_{s}') for s in BASELINES]}
    report['passed']=True;write(out/'verification.json',report);shutil.copy2(__file__,out/'verification_source.py');return report

def worker():
    assert read(PRE/'admission_02/verification.json')['passed'];assert read(PRE/'admission_02/independent_validation.json')['passed']
    folder=DATA/'sitting_ppo_01';folder.mkdir(exist_ok=False);(folder/'frozen').mkdir()
    files=[ROOT/'scripts'/n for n in ['run_repaired_sitting_comparison.py','launch_repaired_sitting_study.py','verify_repaired_sitting.py','run_stable_effort_sitting.py','run_contact_effort_repair.py','run_supported_gpu0.sh']]
    files += [Path(x) for x in read(PRE/'zero_55211/manifest.json')['sources']]
    files += [x for x in (DATA/'banks').rglob('*') if x.is_file()]
    files += [PRE/n for n in ['admission_02/verification.json','admission_02/independent_validation.json','admission_02/definition_revision.json','calibration.json','source_freeze.json']]
    files += [DATA/'sitting_repaired_profile.json',DATA/'plane_calibration.json',ROOT/'docs/REPAIRED_SITTING_EFFORT_PROTOCOL.md']+[Path(parent(s)) for s in PARENTS]
    files += [ROOT/'runs/g1_kinematics_geometry_04/model.json',ROOT/'sitting_task/src/humanoid_sitting/shared_frame.py']
    hashes={str(x):sha(x) for x in files};prior=read(DATA/'prior_preservation.json')['hashes']
    for x in (DATA/'leaning_ppo_01').rglob('*'):
        if x.is_file() and not x.is_symlink() and x.suffix in ['.pt','.json','.npz','.py']:prior[str(x)]=sha(x)
    for i,x in enumerate(hashes):shutil.copy2(x,folder/'frozen'/f'{i:03d}_{Path(x).name}')
    write(folder/'protocol.json',dict(seeds=SEEDS,parent_seeds=PARENTS,updates_per_seed=40,critic_only_updates=8,actor_enabled_updates=32,policy_steps_per_update=16,physics_ticks_per_policy_step=10,num_envs=18,transitions_per_seed=115200,total_transitions=230400,hashes=hashes,prior_hashes=prior,selection='Final checkpoint must improve whole20s and final5s torque versus BOTH repaired reference and repaired own parent while preserving every success and adding no failure; otherwise repaired parent fallback. Holdout after both seeds and selection.'))
    try:
        for seed,oldseed in zip(SEEDS,PARENTS):
            label=f'train_{seed}';status(folder/'status.json','training',active=label);run(folder,label,command(folder,label,'train',seed,parent_cp=parent(oldseed)))
            cp=str(folder/label/'final.pt');label=f'dev_{seed}';status(folder/'status.json','development_evaluation',active=label);run(folder,label,command(folder,label,'evaluate',55501,parent_cp=parent(oldseed),cp=cp))
            gate=comparison(folder,label,PRE/'reference',PRE/f'parent_{oldseed}');write(folder/f'seed_{seed}.json',dict(parent=parent(oldseed),final_checkpoint=cp,gate=gate,parent_fallback=not gate['eligible']))
        write(folder/'selection_frozen.json',dict(unix_time=datetime.now(timezone.utc).timestamp(),selection={str(s):read(folder/f'seed_{s}.json')['parent_fallback'] for s in SEEDS}))
        jobs=[('holdout_old_reference','reference','old',None,None),('holdout_repaired_reference','reference','repaired',None,None)]
        for oldseed in BASELINES:jobs += [(f'holdout_old_parent_{oldseed}','evaluate','old',None,parent(oldseed)),(f'holdout_repaired_parent_{oldseed}','evaluate','repaired',None,parent(oldseed))]
        for seed,oldseed in zip(SEEDS,PARENTS):jobs += [(f'holdout_final_{seed}','evaluate','child',parent(oldseed),str(folder/f'train_{seed}/final.pt'))]
        for label,mode,kind,par,cp in jobs:
            status(folder/'status.json','holdout_evaluation',active=label);run(folder,label,command(folder,label,mode,55501,bank='holdout',kind=kind,parent_cp=par,cp=cp))
        status(folder/'status.json','verifying');verify(folder,folder/'verified_01');status(folder/'status.json','verified_complete')
    except Exception as e:
        status(folder/'status.json','failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',type=Path);p.add_argument('--output',type=Path);a=p.parse_args()
    if a.verify:verify(a.verify,a.output)
    else:worker()
