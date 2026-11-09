"""Independent budgets, gates, optimizer replay and matched final comparisons."""
import argparse,json,shutil
import numpy as np
import torch
from earlier_control_common import *
from earlier_control_gate import gate
from posture_episode_ppo import PosturePPO
from verify_seat_load_study import evaluation,starts,compare,load,equal
from verify_progressive_seat_study import quality
from verify_posture_credit_study import contacts

def expected_reward(rows,baseline):
    return [-(x['metrics_whole_trial']['total_squared_torque_Nm2']-baseline['whole20_Nm2'][i])*20/(9*150)
            -20*any(x['failures'].values())-10*(not x['success'])-5*(x['slip_ticks']>0)-5*(x['illegal_ticks']>0)
            -2*(x['traction_bad_ticks']>baseline['margin_ticks'][i]) for i,x in enumerate(rows)]

def replay(folder,seed,updates):
    cal=read(DATA/f'preflight/calibrate_{seed}/early_calibration.json');initial=load(folder/'initial.pt')
    batches=load(folder/'decision_batches.pt');audit=read(folder/'learning_audit.json');rows=read(folder/'results.json')['episodes']
    assert len(batches)==len(audit)==updates and len(rows)==updates*18
    model=PosturePPO(np.zeros(414),np.ones(414),0);model.load(folder/'initial.pt')
    initial_steps={role:int(next(iter(initial[role]['state'].values()))['step']) if initial[role]['state'] else 0 for role in ['actor_optimizer','critic_optimizer']}
    for i,(b,a) in enumerate(zip(batches,audit)):
        assert b['obs'].shape==(18,414) and b['actions'].shape==(18,3)
        assert np.allclose(expected_reward(rows[18*i:18*(i+1)],cal['baseline']),b['reward'].numpy(),atol=2e-6,rtol=1e-6)
        with torch.no_grad():
            dist=model.distribution(b['obs']);assert torch.allclose(dist.mean,b['old_mean'],atol=2e-5,rtol=2e-5)
            assert torch.allclose(dist.log_prob(b['actions']).sum(-1),b['logp'],atol=1e-4,rtol=2e-5)
        stats=model.update(b['obs'],b['actions'],b['logp'],b['old_mean'],b['reward'])
        assert stats['actor_steps']==a['optimizer']['actor_steps'] and stats['critic_steps']==8 and stats['max_accepted_kl']<=.020001
        saved=load(folder/f'model_{i+1:02d}.pt')
        for role,network in [('actor_state_dict',model.actor),('critic_state_dict',model.critic)]:
            for k,x in network.state_dict().items():assert torch.allclose(x,saved[role][k],atol=5e-5,rtol=5e-4),(folder,i,role,k,float(abs(x-saved[role][k]).max()))
        for role,field in [('actor_optimizer','actor_steps'),('critic_optimizer','critic_steps')]:
            expected=initial_steps[role]+sum(x['optimizer'][field] for x in audit[:i+1])
            assert all(int(x['step'])==expected for x in saved[role]['state'].values())
        assert saved['updates']==initial['updates']+i+1
        assert np.allclose(a['offsets_rad'],EARLY_SPAN*np.tanh(b['actions'].numpy()),atol=1e-8)
    final=load(folder/'final.pt');assert equal(final,load(folder/f'model_{updates:02d}.pt'))
    for role in ['actor_state_dict','critic_state_dict']:
        assert all(torch.isfinite(x).all() for x in final[role].values())
        assert any(not torch.equal(x,final[role][k]) for k,x in initial[role].items())
    return dict(passed=True,CPU_replay=True,weight_atol=5e-5,weight_rtol=5e-4,actor_steps=sum(x['optimizer']['actor_steps'] for x in audit),critic_steps=8*updates)

def paths(study,bank):
    if bank=='development':
        result={'reference':OLD/'preflight/calibrate',**{f'parent_{PARENTS[s]}':OLD/f'ppo_01/seed_{PARENTS[s]}/development_final' for s in SEEDS}}
    elif bank=='retention':
        result={'reference':OLD/'ppo_01/holdout_reference',**{f'parent_{PARENTS[s]}':OLD/f'ppo_01/holdout_policy_{PARENTS[s]}' for s in SEEDS}}
    else:result={'reference':study/'wide_reference',**{f'parent_{PARENTS[s]}':study/f'wide_parent_{PARENTS[s]}' for s in SEEDS}}
    for s in SEEDS:
        final=read(study/f'training_finished_{s}.json');selected=read(study/f'seed_{s}/selection.json')
        result[f'final_{s}']=Path(final['development_evaluation']) if bank=='development' else study/f'{bank}_final_{s}'
        if not selected.get('parent_fallback') and (selected['checkpoint'],selected['start_s'])!=(final['checkpoint'],final['start_s']):
            result[f'selected_{s}']=Path(selected['evaluation']) if bank=='development' else study/f'{bank}_selected_{s}'
    return result

def source_check(folder,protocol):
    m=read(folder/'manifest.json')
    for p,h in m['sources'].items():assert protocol['hashes'][p]==h,(folder,p)
    for name in ['run_earlier_control_ppo.py','earlier_control_common.py','run_posture_episode_ppo.py','posture_episode_ppo.py','run_traction_transfer_ppo.py']:
        assert sha(folder/'source'/name)==protocol['hashes'][str(ROOT/'scripts'/name)]
    for name,h in m['bank_hashes'].items():assert sha(Path(m['args']['bank'])/name)==h
    assert read(folder/'traction_config.json')==read(DATA.parent/'traction_transfer_01/selected_controller.json')

def matched_start(reference,folder):
    result=starts(reference,folder);assert result['exact']
    a=np.load(reference/'traction_startup.npz');b=np.load(folder/'traction_startup.npz')
    for key in a.files:assert np.array_equal(a[key],b[key]),(folder,'actual 414 startup',key)
    assert a['policy'].shape[-1]==414
    return dict(**result,actual_414channel_startup_exact=True)

def actor_checks(folder,seed,checkpoint,start):
    cal=read(DATA/f'preflight/calibrate_{seed}/early_calibration.json');norm=cal['stages'][str(start)]
    initial=load(folder/'initial.pt');manifest=read(folder/'manifest.json');parent=parent_checkpoint(seed)
    assert manifest['parent_checkpoint_sha256']==sha(parent) and equal(load(parent),load(folder/'loaded_parent.pt'))
    if checkpoint:
        assert manifest['early_checkpoint_sha256']==sha(checkpoint) and sha(folder/'loaded_early.pt')==sha(checkpoint)
        saved=load(checkpoint)
        for role in ['actor_state_dict','critic_state_dict','actor_optimizer','critic_optimizer']:assert equal(initial[role],saved[role])
        assert initial['updates']==saved['updates']
    assert torch.equal(initial['input_mean'],torch.tensor(norm['mean'],dtype=torch.float32))
    assert torch.equal(initial['input_scale'],torch.tensor(norm['scale'],dtype=torch.float32))
    d=np.load(folder/'episode_decisions.npz');policy=PosturePPO(np.zeros(414),np.ones(414),0);policy.load(folder/'initial.pt')
    frozen=PosturePPO(np.zeros(414),np.ones(414),0);frozen.load(parent)
    contract=read(folder/'early_contract.json')
    assert contract['start_s']==start and contract['handover_s']==11 and contract['early_span']==EARLY_SPAN and contract['parent_span']==.008
    assert d['early_observation'].shape[-1]==d['parent_observation'].shape[-1]==414
    with torch.no_grad():
        for i,obs in enumerate(d['parent_observation']):
            actual=frozen.choose(torch.tensor(obs),False)[0].numpy();assert np.allclose(actual,d['parent_action'][i],atol=1e-5,rtol=1e-5)
        if contract['mode']=='evaluate':
            actual=policy.choose(torch.tensor(d['early_observation'][0]),False)[0].numpy();assert np.allclose(actual,d['early_action'][0],atol=1e-5,rtol=1e-5)
    assert abs(EARLY_SPAN*np.tanh(d['early_action'])).max()<=EARLY_SPAN and abs(.008*np.tanh(d['parent_action'])).max()<=.008
    return dict(parent_frozen_exact=True,loaded_early_exact=True,normalization_exact=True,actual_414channel_outputs_verified=True,early_bound_rad=EARLY_SPAN,parent_bound_rad=.008)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    protocol=read(a.study/'protocol.json');assert sha(a.study/'protocol.json')==(a.study/'protocol.sha256').read_text().strip();check(protocol['hashes']);check(protocol['prior_hashes'])
    barrier=read(a.study/'holdout_barrier.json');out=dict(passed=False,source_hashes=len(protocol['hashes']),prior_hashes=len(protocol['prior_hashes']),training={},evaluations={},contacts={},startups={},actors={},comparisons={},selections={},paths={})
    for seed in SEEDS:
        folder=a.study/f'seed_{seed}';records=read(folder/'curriculum.json');assert len(records)==4;start=10.;previous=None;candidates=[];total=dict(active=0,warmup=0,initial_settle=0,actor_steps=0,critic_steps=0,stable=0,physical=0,slip=0,illegal=0);stage_updates={}
        for i,row in enumerate(records):
            assert row['block']==i and row['trained_start_s']==start;train=folder/f'block_{i:02d}/train';c=read(train/'early_contract.json');assert c['active']==108000 and c['warmup']==2250 and c['total']==110250 and c['episodes']==108 and c['initial_settle_excluded']==450 and c['local_updates']==6
            source_check(train,protocol);actor_checks(train,seed,previous,start)
            accounting=read(train/'control_accounting.json');assert accounting['excluded_ticks']==0
            assert accounting['control_ticks']==int((20-start)/.02)*108 and accounting['teacher_ticks']==int(start/.02)*108
            assert accounting['active_physics']==108000 and accounting['indexed_warmup']==2250
            assert all(abs(t-start)<1e-6 for t in accounting['earliest_control_s'])
            if i==0:assert equal(load(train/'initial.pt'),load(PREFLIGHT/f'smoke_{seed}/initial.pt'))
            replay_result=replay(train,seed,6)
            for key in ['active','warmup']:total[key]+=c[key]
            total['initial_settle']+=450
            for key in ['actor_steps','critic_steps']:total[key]+=replay_result[key]
            audit=read(train/'learning_audit.json')
            for key in ['stable','physical','slip','illegal']:total[key]+=sum(x[key] for x in audit)
            stage_updates[str(start)]=stage_updates.get(str(start),0)+6
            cp=train/'final.pt';assert str(cp)==row['checkpoint'] and sha(cp)==row['sha256'];g=gate(Path(row['current_evaluation']),seed);assert g==row['current_gate']
            for label,t in [('current_evaluation',start)]+([('earlier_evaluation',row['proposed_start_s'])] if 'proposed_start_s' in row else []):
                tested=Path(row[label]);source_check(tested,protocol);actor_checks(tested,seed,cp,t);evaluation(tested,18);contacts(tested);matched_start(OLD/'preflight/calibrate',tested)
            if g['passed']:candidates.append(dict(checkpoint=str(cp),sha256=sha(cp),start_s=start,evaluation=row['current_evaluation'],gate=g,block=i))
            if 'proposed_start_s' in row:
                assert i<3 and g['passed'] and row['proposed_start_s']==STAGES[STAGES.index(start)+1]
                nxt=gate(Path(row['earlier_evaluation']),seed);assert nxt==row['earlier_gate']
                if nxt['passed']:
                    start=row['proposed_start_s'];candidates.append(dict(checkpoint=str(cp),sha256=sha(cp),start_s=start,evaluation=row['earlier_evaluation'],gate=nxt,block=i))
            previous=cp
        assert total['active']==432000 and total['warmup']==9000 and total['initial_settle']==1800
        final=read(a.study/f'training_finished_{seed}.json');assert final==barrier['finals'][str(seed)] and final['sha256']==sha(previous) and final['timestamp']<barrier['timestamp']
        selected=read(folder/'selection.json');assert selected==barrier['selections'][str(seed)] and selected['selection_timestamp']<barrier['timestamp']
        if candidates:
            expected=min(candidates,key=lambda x:(x['start_s'],x['gate']['effort']['metrics_whole_trial']['policy'],x['gate']['effort']['metrics_last5s']['policy']))
            for k,v in expected.items():assert selected[k]==v
        else:assert selected['parent_fallback'] and selected['sha256']==sha(parent_checkpoint(seed))
        out['training'][str(seed)]=dict(**total,transitions=total['active']+total['warmup'],stage_updates=stage_updates,final_start_s=final['start_s'],optimizer_replay=True,normalization_changes_predefined=True)
        out['selections'][str(seed)]=selected
    for bank,count in [('development',18),('retention',16),('wide',32)]:
        all_paths=paths(a.study,bank);out['paths'][bank]={k:str(v) for k,v in all_paths.items()};all_rows={};geoms={}
        for name,folder in all_paths.items():
            summary,rows,g=evaluation(folder,count);quality(summary,rows);key=bank+'_'+name
            out['evaluations'][key]=summary;out['contacts'][key]=contacts(folder);out['startups'][key]=matched_start(all_paths['reference'],folder)
            all_rows[name]=rows;geoms[name]=g
            if name.startswith(('final_','selected_')):
                seed=int(name.split('_')[1]);which=read(a.study/f'training_finished_{seed}.json') if name.startswith('final') else read(a.study/f'seed_{seed}/selection.json')
                source_check(folder,protocol);out['actors'][key]=actor_checks(folder,seed,Path(which['checkpoint']),which['start_s'])
                if bank=='development':
                    recorded=np.load(folder/'episode_decisions.npz')['early_observation'][0]
                    matched=np.load(DATA/f'preflight/calibrate_{seed}/stage_observations.npz')[str(which['start_s'])][0]
                    assert np.array_equal(recorded,matched),'Actual early inputs must match parent before any action'
                x=np.load(all_paths['reference']/'trace.npz');y=np.load(folder/'trace.npz');mask=x['t']<=which['start_s']
                for field in ['q','body','command','reference','torque','geometry']:assert np.array_equal(x[field][mask],y[field][mask]),(key,field,'prefix')
                if bank!='development':assert (folder/'manifest.json').stat().st_mtime>barrier['timestamp']
                out['startups'][key]['prefix_exact_until_s']=which['start_s']
            if bank=='wide':assert (folder/'manifest.json').stat().st_mtime>barrier['timestamp']
        for name in all_rows:
            if name.startswith(('final_','selected_')):
                clean=lambda rows:[dict(x,success=bool(x['success'] and not any(x['failures'].values()) and x['slip_ticks']==0 and x['illegal_ticks']==0)) for x in rows]
                out['comparisons'][bank+'_'+name]={b:compare(clean(all_rows[name]),clean(all_rows[b]),geoms[name]) for b in ['reference','parent_59111','parent_59117']}
                out['comparisons'][bank+'_'+name+'_strict']={b:compare([x for x in all_rows[name] if x['strict_transfer']],clean(all_rows[b]),geoms[name]) for b in ['reference','parent_59111','parent_59117']}
    out['comparison_scope']='Common full-duration stable successes with zero all-tick slip or unwanted contact in BOTH policy and baseline. Strict subsets additionally require policy margin-clean.'
    check(protocol['hashes']);check(protocol['prior_hashes']);out['passed']=True;write(a.output/'verification.json',out);shutil.copy2(__file__,a.output/'verification_source.py')
    print(json.dumps(dict(passed=True,training=out['training'],outcomes={k:(v['successes'],v['trials']) for k,v in out['evaluations'].items()}),indent=2))
if __name__=='__main__':main()
