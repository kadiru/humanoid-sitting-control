"""Independent audit of the restricted-domain, progressive-control PPO pilot."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from verify_seat_load_study import equal,load,evaluation,starts,compare
from verify_progressive_seat_study import quality
from run_crouch_rest_safe_study import DATA,ROOT,ELIGIBLE,read,sha,check,gate

def bridge_sources(path,protocol):
    for name,h in read(path/'bridge_sources.json').items():assert protocol['hashes'][name]==h,(path,name)
    for name in ['run_crouch_rest_audited_ppo.py','run_crouch_rest_safe_ppo.py','crouch_rest_validated_controller.py','crouch_rest_motion.py','sagittal_static_bias.py']:
        assert sha(path/'source'/name)==protocol['hashes'][str(ROOT/'scripts'/name)]

def matched(reference,path,start):
    result=starts(reference,path);assert result['exact'],result
    for file in ['bridge_startup.npz']:
        a=np.load(reference/file);b=np.load(path/file);assert all(np.array_equal(a[k],b[k]) for k in a.files)
    a=np.load(reference/'trace.npz');b=np.load(path/'trace.npz');mask=a['t']<=start
    for k in ['q','body','command','reference','torque','geometry']:assert np.array_equal(a[k][mask],b[k][mask]),(path,k,start)
    result.update(actual_actor_startup_exact=True,teacher_prefix_exact_through_s=start)
    return result

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    protocol=read(a.study/'protocol.json');assert sha(a.study/'protocol.json')==(a.study/'protocol.sha256').read_text().strip();check(protocol);barrier=read(a.study/'holdout_barrier.json')
    reference=Path(protocol['bridge_reference']);out=dict(passed=False,source_hashes=len(protocol['hashes']),prior_hashes=len(protocol['prior_hashes']),training={},evaluations={},startups={},comparisons={},reference_repair_comparisons={},caveats=['Six of nine development geometries admitted; three remain reported stress tests.','All resets start standing, but learned control starts at10–15s and the first10s stay reference-controlled.','Two seeds and repeated rigid-surface placements do not establish arbitrary-surface generalization.','Torque is an effort proxy, not electrical energy or a global optimum.','Scheduled rollout updates can contain no learning samples; actual Adam steps are reported separately.','All-tick failure/slip counts use preserved runtime labels; saved motion/command samples are100ms.'])
    finals={}
    for seed in protocol['seeds']:
        folder=a.study/f'seed_{seed}';records=read(folder/'curriculum.json');assert len(records)==4
        start=15;previous=None;initial_first=None;totals={k:0 for k in ['updates','actor_updates','policy_decisions','transitions','active','warmup','control_ticks','teacher_ticks','excluded_ticks','initial_settle_ticks_excluded']};blocks=[]
        for i,record in enumerate(records):
            assert record['block']==i and record['updates']==16 and record['learning_start_s']==start
            path=Path(record['checkpoint']).parent;v=read(path/'verification.json');c=read(path/'control_accounting.json');contract=read(path/'bridge_contract.json');bridge_sources(path,protocol)
            assert contract['control_window_start_s']==start and read(path/'manifest.json')['args']['curriculum_stage']==0
            assert v['updates']==16 and v['actor_updates']==16-(8 if i==0 else 0) and v['policy_decisions']==9216 and v['transitions']==46080 and v['active']+v['warmup']==46080
            assert c['active_physics']==v['active']==c['control_ticks']+c['teacher_ticks']+c['excluded_ticks'] and c['indexed_warmup']==v['warmup']
            assert sum(c['control_by_case'])==c['control_ticks'] and all(c['control_by_case'][j]==0 for j in [6,7,12,13,14,15])
            assert all(t>=start for t in c['earliest_control_s'])
            initial=load(path/'initial.pt');final=load(path/'final.pt')
            if previous is not None:
                assert all(equal(initial[k],previous[k]) for k in ['actor_state_dict','critic_state_dict','optimizer_state_dict']);assert initial['iter']==previous['iter']+1
            else:
                initial_first=initial;warm=load(path/'critic_warmup.pt');assert equal(initial['actor_state_dict'],warm['actor_state_dict']);assert set(warm['optimizer_state_dict']['state'])==set(range(7,13))
                assert initial['actor_state_dict']['mlp.0.weight'].shape==(64,406) and initial['actor_state_dict']['mlp.4.weight'].shape==(23,64)
                assert initial['actor_state_dict']['mlp.4.weight'].count_nonzero()==0 and initial['actor_state_dict']['mlp.4.bias'].count_nonzero()==0
            for role in ['actor_state_dict','critic_state_dict']:assert all(torch.isfinite(x).all() for x in final[role].values())
            for k in ['updates','actor_updates','policy_decisions','transitions','active','warmup']:totals[k]+=v[k]
            for k in ['control_ticks','teacher_ticks','excluded_ticks']:totals[k]+=c[k]
            opt=final['optimizer_state_dict'];actor_steps=[int(opt['state'][j]['step']) for j in range(7)];critic_steps=[int(opt['state'][j]['step']) for j in range(7,13)]
            assert len(set(actor_steps))==len(set(critic_steps))==1 and 0<actor_steps[0]<=16*totals['actor_updates'] and actor_steps[0]<=critic_steps[0]<=16*totals['updates'];assert opt['param_groups'][0]['lr']==1.e-4
            assert final['iter']==totals['updates']-1
            episodes=read(path/'results.json')['episodes'];assert all(x['start_stage']==0 and x['warm_failure_ticks']==0 for x in episodes)
            pending=25*len(episodes)-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in episodes);assert 0<=pending<=450 and 0<=unfinished<18000
            settling=read(path/'initial_settling.json');assert len(settling)==25 and not any(any(z) for row in settling for z in row['failures'].values());totals['initial_settle_ticks_excluded']+=450
            evalpath=Path(record['evaluation']);summary,rows,g=evaluation(evalpath,18,Path(record['checkpoint']));quality(summary,rows);bridge_sources(evalpath,protocol);assert gate(evalpath)==record['current_gate'];matched(reference,evalpath,start)
            next_start=start;proposal={15:13,13:11,11:10,10:10}[start]
            if i<3 and record['current_gate']['passed'] and proposal<start:
                early=folder/f'advance_{i:02d}_control{proposal}s';s,rr,gg=evaluation(early,18,Path(record['checkpoint']));quality(s,rr);assert gate(early)==record['earlier_gate'];matched(reference,early,proposal);bridge_sources(early,protocol)
                if record['earlier_gate']['passed']:next_start=proposal
            else:assert record['earlier_gate'] is None
            assert record['next_start_s']==next_start;blocks.append(dict(block=i,learning_start_s=start,**v,control_accounting=c,pending_warmup_ticks=pending,unfinished_active_ticks=unfinished,actual_actor_Adam_steps=actor_steps[0],actual_critic_Adam_steps=critic_steps[0],gate=record['current_gate'],earlier_gate=record['earlier_gate'],next_start_s=next_start));start=next_start;previous=final
        assert totals['updates']==64 and totals['actor_updates']==56 and totals['transitions']==184320 and totals['initial_settle_ticks_excluded']==1800
        assert totals['active']==totals['control_ticks']+totals['teacher_ticks']+totals['excluded_ticks']
        for role in ['actor_state_dict','critic_state_dict']:assert any(not torch.equal(initial_first[role][k],v) for k,v in previous[role].items())
        finished=read(a.study/f'training_finished_{seed}.json');cp=Path(finished['checkpoint']);assert sha(cp)==finished['sha256']==records[-1]['checkpoint_sha256'];assert finished['timestamp']<=barrier['timestamp'] and barrier['finals'][str(seed)]==finished
        finals[str(seed)]=finished;out['training'][str(seed)]=dict(**totals,all_resets_standing=True,final_control_start_s=records[-1]['learning_start_s'],actual_actor_Adam_steps=actor_steps[0],actual_critic_Adam_steps=critic_steps[0],blocks=blocks)
    assert sum(x['transitions'] for x in out['training'].values())==368640
    for bank,count in [('development',18),('holdout',16)]:
        paths={k:Path(v) for k,v in protocol['development_baselines'].items()} if bank=='development' else {k:a.study/('holdout_'+k) for k in ['current_reference','current_56211','current_56217']}
        paths['bridge_reference']=reference if bank=='development' else a.study/'holdout_bridge_reference'
        for seed,f in finals.items():paths[seed]=Path(f['final_development_evaluation']) if bank=='development' else a.study/f'holdout_policy_{seed}'
        rr={};gg={}
        for label,path in paths.items():
            cp=Path(finals[label]['checkpoint']) if label in finals else DATA.parent/'contact_effort_repair_01'/f"sitting_ppo_01/train_{label.split('_')[1]}/final.pt" if label.startswith('current_56') else None
            summary,rows,g=evaluation(path,count,cp);quality(summary,rows);startup=starts(paths['current_reference'],path);assert startup['exact'],startup
            if label.startswith('current_56'):
                parent=read(path/'parent_identity.json');assert sha(parent['path'])==parent['sha256']
            if label in finals:
                startup.update(matched(paths['bridge_reference'],path,finals[label]['learning_start_s']));bridge_sources(path,protocol)
            if bank=='development':
                summary['admitted']={k:sum((x['success'] if k=='stable' else x.get('strict_transfer',False) if k=='strict' else any(x['failures'].values())) for x in rows if x['geometry_index'] in ELIGIBLE) for k in ['stable','strict','physical']};summary['admitted']['trials']=12
                summary['excluded']={k:sum((x['success'] if k=='stable' else x.get('strict_transfer',False) if k=='strict' else any(x['failures'].values())) for x in rows if x['geometry_index'] not in ELIGIBLE) for k in ['stable','strict','physical']};summary['excluded']['trials']=6
                if 'strict_transfer' not in rows[0]:summary['admitted']['strict']=None;summary['excluded']['strict']=None
            out['evaluations'][f'{bank}_{label}']=summary;out['startups'][f'{bank}_{label}']=startup;rr[label]=rows;gg[label]=g
            if bank=='holdout':assert (path/'manifest.json').stat().st_mtime>=barrier['timestamp']
        for seed in finals:
            out['comparisons'][f'{bank}_{seed}']={b:compare(rr[seed],rr[b],gg[seed]) for b in ['current_reference','current_56211','current_56217','bridge_reference']}
            out['comparisons'][f'{bank}_{seed}_strict_policy']={b:compare([x for x in rr[seed] if x['strict_transfer']],rr[b],gg[seed]) for b in ['current_reference','current_56211','current_56217','bridge_reference']}
        out['reference_repair_comparisons'][bank]={b:compare(rr['bridge_reference'],rr[b],gg['bridge_reference']) for b in ['current_reference','current_56211','current_56217']}
    check(protocol);out['passed']=True;(a.output/'verification.json').write_text(json.dumps(out,indent=2));shutil.copy2(__file__,a.output/'verification_source.py');print(json.dumps(dict(passed=True,training={s:{k:v for k,v in x.items() if k!='blocks'} for s,x in out['training'].items()}),indent=2))
if __name__=='__main__':main()
