"""Independent execution, curriculum, matched-evaluation and effort audit."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from verify_seat_load_study import equal,load,evaluation,starts,compare
from run_progressive_seat_study import DATA,OLD,read,sha,check

def quality(summary,rows):
    if 'strict_transfer' not in rows[0]:return summary
    for x in rows:
        expected=x['success'] and x['slip_ticks']==0 and x['traction_bad_ticks']==0 and x['illegal_ticks']==0
        assert x['strict_transfer']==expected
        assert all(0<=x[k]<=1000 for k in ['slip_ticks','traction_bad_ticks','illegal_ticks'])
        assert x['warm_failure_ticks']==0
    summary.update(strict_successes=sum(x['strict_transfer'] for x in rows),slip_cases=[x['case'] for x in rows if x['slip_ticks']],friction_margin_cases=[x['case'] for x in rows if x['traction_bad_ticks']])
    return summary

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    protocol=read(a.study/'protocol.json');check(protocol);barrier=read(a.study/'holdout_barrier.json')
    out=dict(passed=False,source_hashes=len(protocol['hashes']),prior_hashes=len(protocol['prior_hashes']),training={},curriculum={},evaluations={},startup={},comparisons={},strong_rest_goal={},caveats=['Two seeds; repeated placements on finite rigid surfaces.','Near-seat and standing success remain separate.','Torque is an effort proxy, not energy or a global minimum.','All-tick physical/slip/friction summaries use frozen runtime labels; saved command traces are100ms samples.','Curriculum gates use development only; full-budget final checkpoints are the comparison, with no automatic promotion.'])
    checkpoints={}
    for seed in protocol['seeds']:
        folder=a.study/f'seed_{seed}';curriculum=read(folder/'curriculum.json');assert len(curriculum)==8
        totals=dict(updates=0,actor_updates=0,policy_decisions=0,transitions=0,active=0,warmup=0,initial_settle_ticks_excluded=0);stage=5;previous=None;first=None;blocks=[];stages={}
        for i,(record,updates) in enumerate(zip(curriculum,protocol['block_updates'])):
            assert record['block']==i and record['stage']==stage and record['updates']==updates
            path=Path(record['checkpoint']).parent;v=read(path/'verification.json');manifest=read(path/'manifest.json');initial=load(path/'initial.pt');final=load(path/'final.pt')
            assert manifest['args']['curriculum_stage']==stage and manifest['args']['iterations']==updates
            assert v['updates']==updates and v['actor_updates']==updates-(8 if i==0 else 0)
            assert v['policy_decisions']==updates*32*18 and v['transitions']==updates*32*5*18 and v['active']+v['warmup']==v['transitions']
            if previous is not None:
                for k in ['actor_state_dict','critic_state_dict','optimizer_state_dict']:assert equal(initial[k],previous[k]),(seed,i,k)
                assert initial['iter']==previous['iter']+1
            else:
                first=initial;warm=load(path/'critic_warmup.pt');assert equal(initial['actor_state_dict'],warm['actor_state_dict']);assert set(warm['optimizer_state_dict']['state'])==set(range(7,13))
                assert initial['actor_state_dict']['mlp.0.weight'].shape==(64,406) and initial['actor_state_dict']['mlp.4.weight'].shape==(23,64)
                assert initial['actor_state_dict']['mlp.4.weight'].count_nonzero()==0 and initial['actor_state_dict']['mlp.4.bias'].count_nonzero()==0
            for role in ['actor_state_dict','critic_state_dict']:assert all(torch.isfinite(x).all() for x in final[role].values())
            for k in ['updates','actor_updates','policy_decisions','transitions','active','warmup']:totals[k]+=v[k]
            assert final['iter']==totals['updates']-1
            opt=final['optimizer_state_dict'];assert all(int(opt['state'][j]['step'])==16*totals['actor_updates'] for j in range(7));assert all(int(opt['state'][j]['step'])==16*totals['updates'] for j in range(7,13));assert opt['param_groups'][0]['lr']==1e-4
            episodes=read(path/'results.json')['episodes'];assert all(x['start_stage']==stage and x['warm_failure_ticks']==0 for x in episodes)
            pending=25*len(episodes)-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in episodes);assert 0<=pending<=450 and 0<=unfinished<18000
            settling=read(path/'initial_settling.json');assert len(settling)==25 and not any(any(z) for row in settling for z in row['failures'].values());totals['initial_settle_ticks_excluded']+=450
            gatepath=Path(record['evaluation']);summary,rows,g=evaluation(gatepath,18,Path(record['checkpoint']));quality(summary,rows)
            assert all(x['start_stage']==stage for x in rows)
            loaded=sum(x['metrics_last5s']['vertical_support_load_bw']>=.6 for x in rows)
            gate=dict(passed=summary['physical_failures']==0 and summary['successes']==18 and summary['strict_successes']==18 and loaded==18,physical=summary['physical_failures'],stable=summary['successes'],strict=summary['strict_successes'],seat_majority=loaded,trials=18)
            assert gate==record['gate'];next_stage=max(0,stage-1) if gate['passed'] else stage;assert record['next_stage']==next_stage
            stages[str(stage)]=stages.get(str(stage),0)+v['transitions'];blocks.append(dict(block=i,stage=stage,**v,pending_warmup_ticks=pending,unfinished_active_ticks=unfinished,gate=gate));stage=next_stage;previous=final
        assert totals['updates']==128 and totals['actor_updates']==120 and totals['transitions']==368640 and totals['initial_settle_ticks_excluded']==3600
        for role in ['actor_state_dict','critic_state_dict']:assert any(not torch.equal(first[role][k],v) for k,v in previous[role].items())
        cp=Path(curriculum[-1]['checkpoint']);assert sha(cp)==curriculum[-1]['checkpoint_sha256'];done=read(a.study/f'training_finished_{seed}.json');assert done['sha256']==sha(cp) and done['time']<=barrier['time'];assert barrier['checkpoints'][str(seed)]==done
        checkpoints[str(seed)]=cp;out['training'][str(seed)]=dict(**totals,transitions_by_stage=stages,standing_transitions=stages.get('0',0),last_trained_stage=curriculum[-1]['stage'],next_stage=stage,blocks=blocks);out['curriculum'][str(seed)]=curriculum
    assert sum(x['transitions'] for x in out['training'].values())==737280
    oldprotocol=read(DATA.parent/'seat_load_transfer_01/ppo_01/protocol.json')
    for bank in ['development','holdout']:
        if bank=='development':
            paths={k:Path(v) for k,v in oldprotocol['development_baselines'].items() if k!='new_reference'};paths['new_reference']=DATA/'harness_standing_01'
            for seed in protocol['seeds']:paths[str(seed)]=a.study/f'seed_{seed}/final_development_standing'
        else:
            paths={k:a.study/('holdout_'+k) for k in ['current_reference','current_56211','current_56217','new_reference']}
            for seed in protocol['seeds']:paths[str(seed)]=a.study/f'holdout_policy_{seed}'
        rr={};gg={}
        for label,path in paths.items():
            cp=checkpoints.get(label)
            if label in ['current_56211','current_56217']:
                cp=OLD/f"sitting_ppo_01/train_{label.split('_')[1]}/final.pt";parent=read(path/'parent_identity.json');assert parent['sha256']==sha(parent['path'])
            summary,rows,g=evaluation(path,18 if bank=='development' else 16,cp);quality(summary,rows)
            assert all(x['start_stage']==0 for x in rows)
            out['evaluations'][f'{bank}_{label}']=summary;out['startup'][f'{bank}_{label}']=starts(paths['current_reference'],path);rr[label]=rows;gg[label]=g
            if bank=='holdout':assert (path/'manifest.json').stat().st_mtime>=barrier['time']
        for seed in protocol['seeds']:
            label=str(seed);out['comparisons'][f'{bank}_{seed}']={b:compare(rr[label],rr[b],gg[label]) for b in ['current_reference','current_56211','current_56217','new_reference']}
            base={x['case']:x for x in rr['current_reference']};cases=[x['case'] for x in rr[label] if x['success'] and x['metrics_last5s']['vertical_support_load_bw']>=.5 and x['metrics_last5s']['total_squared_torque_Nm2']<=.35*base[x['case']]['metrics_last5s']['total_squared_torque_Nm2']]
            out['strong_rest_goal'][f'{bank}_{seed}']=dict(count=len(cases),cases=cases,definition='Full stable success, final5s seat vertical load>=.5BW and torque²<=35% of matched current reference; paired effort still restricted to common successes.')
    nearbase=DATA/'harness_near_01';summary,reference,g=evaluation(nearbase,18);out['evaluations']['near_reference']=quality(summary,reference)
    for seed in protocol['seeds']:
        path=a.study/f'seed_{seed}/final_development_near';summary,rows,g=evaluation(path,18,checkpoints[str(seed)]);assert all(x['start_stage']==5 for x in rows)
        out['evaluations'][f'near_{seed}']=quality(summary,rows);out['startup'][f'near_{seed}']=starts(nearbase,path);out['comparisons'][f'near_{seed}']={'near_reference':compare(rows,reference,g)}
    check(protocol);out['passed']=True;(a.output/'verification.json').write_text(json.dumps(out,indent=2));shutil.copy2(__file__,a.output/'verification_source.py');print(json.dumps({'passed':True,'training':{s:{k:v for k,v in x.items() if k!='blocks'} for s,x in out['training'].items()},'evaluations':{k:{z:v[z] for z in ['successes','physical_failures','trials']} for k,v in out['evaluations'].items()}},indent=2))
if __name__=='__main__':main()
