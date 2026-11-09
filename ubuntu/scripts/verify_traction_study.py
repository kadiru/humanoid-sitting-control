"""Independent budget, checkpoint, sensor/startup and paired-outcome audit."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
import torch
from verify_seat_load_study import equal,load,evaluation,starts,compare
from verify_progressive_seat_study import quality
from run_traction_study import ROOT,read,sha,check

def source_check(path,protocol):
    manifest=read(path/'manifest.json')
    for name,h in manifest['sources'].items():assert protocol['hashes'][name]==h,(path,name)
    for name in protocol['new_sources']:
        assert sha(path/'source'/name)==protocol['hashes'][str(ROOT/'scripts'/name)],(path,name)
    assert read(path/'traction_config.json')==protocol['controller']

def matched(reference,path,until=11):
    result=starts(reference,path);assert result['exact'],result
    a=np.load(reference/'traction_startup.npz');b=np.load(path/'traction_startup.npz')
    assert all(np.array_equal(a[k],b[k]) for k in a.files)
    a=np.load(reference/'trace.npz');b=np.load(path/'trace.npz');mask=a['t']<=until
    for k in ['q','body','command','reference','torque','geometry']:assert np.array_equal(a[k][mask],b[k][mask]),(path,k)
    return dict(**result,actor_startup_exact=True,reference_prefix_exact_through_s=until)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    protocol=read(a.study/'protocol.json');assert sha(a.study/'protocol.json')==(a.study/'protocol.sha256').read_text().strip();assert protocol['admitted'];check(protocol)
    barrier=read(a.study/'holdout_barrier.json');finals={};out=dict(passed=False,source_hashes=len(protocol['hashes']),prior_hashes=len(protocol['prior_hashes']),training={},evaluations={},startups={},comparisons={},reference_repair_comparisons={},caveats=protocol['caveats'])
    for seed in protocol['seeds']:
        folder=a.study/f'seed_{seed}/train';v=read(folder/'verification.json');c=read(folder/'control_accounting.json');source_check(folder,protocol)
        assert v['updates']==64 and v['actor_updates']==56 and v['policy_decisions']==36864 and v['transitions']==184320
        assert v['active']+v['warmup']==184320 and c['active_physics']==v['active']==c['control_ticks']+c['teacher_ticks'] and c['excluded_ticks']==0
        assert c['indexed_warmup']==v['warmup'] and sum(c['control_by_case'])==c['control_ticks'] and all(t>=11 for t in c['earliest_control_s'])
        initial=load(folder/'initial.pt');warm=load(folder/'critic_warmup.pt');final=load(folder/'final.pt')
        assert equal(initial['actor_state_dict'],warm['actor_state_dict']) and final['iter']==63
        assert initial['actor_state_dict']['mlp.0.weight'].shape==(64,414) and initial['actor_state_dict']['mlp.4.weight'].shape==(23,64)
        assert initial['actor_state_dict']['mlp.4.weight'].count_nonzero()==0 and initial['actor_state_dict']['mlp.4.bias'].count_nonzero()==0
        for role in ['actor_state_dict','critic_state_dict']:
            assert all(torch.isfinite(x).all() for x in final[role].values())
            assert any(not torch.equal(x,final[role][k]) for k,x in initial[role].items())
        assert set(warm['optimizer_state_dict']['state'])==set(range(7,13))
        opt=final['optimizer_state_dict'];actor_steps=[int(opt['state'][j]['step']) for j in range(7)];critic_steps=[int(opt['state'][j]['step']) for j in range(7,13)]
        assert len(set(actor_steps))==len(set(critic_steps))==1 and 0<actor_steps[0]<=896 and actor_steps[0]<=critic_steps[0]<=1024
        assert opt['param_groups'][0]['lr']==1.e-4
        episodes=read(folder/'results.json')['episodes'];assert all(x['start_stage']==0 and x['warm_failure_ticks']==0 for x in episodes)
        pending=25*len(episodes)-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in episodes);assert 0<=pending<=450 and 0<=unfinished<18000
        settling=read(folder/'initial_settling.json');assert len(settling)==25 and not any(any(z) for row in settling for z in row['failures'].values())
        finished=read(a.study/f'training_finished_{seed}.json');assert finished['sha256']==sha(folder/'final.pt') and finished['timestamp']<=barrier['timestamp'] and barrier['finals'][str(seed)]==finished
        finals[str(seed)]=finished
        out['training'][str(seed)]=dict(**v,control_accounting=c,pending_warmup_ticks=pending,unfinished_active_ticks=unfinished,initial_settle_ticks_excluded=450,actual_actor_Adam_steps=actor_steps[0],actual_critic_Adam_steps=critic_steps[0],all_resets_standing=True,learned_control_starts_s=11,optimizer_continuity='One uninterrupted training process per seed; no block restart')
    assert sum(x['transitions'] for x in out['training'].values())==368640
    for bank,count in [('development',18),('holdout',16)]:
        paths={k:Path(v) for k,v in protocol['development_baselines'].items()} if bank=='development' else {k:a.study/('holdout_'+k) for k in protocol['development_baselines']}
        for seed in finals:paths[seed]=a.study/(f'seed_{seed}/development_final' if bank=='development' else f'holdout_policy_{seed}')
        rr={};gg={}
        for name,path in paths.items():
            checkpoint=Path(finals[name]['checkpoint']) if name in finals else Path(protocol['baseline_checkpoints'][name]) if name in protocol['baseline_checkpoints'] else None
            summary,rows,g=evaluation(path,count,checkpoint);quality(summary,rows);assert all(x['start_stage']==0 for x in rows)
            startup=starts(paths['current_reference'],path);assert startup['exact'],(name,startup)
            if name in finals:
                startup.update(matched(paths['new_reference'],path));source_check(path,protocol)
            if name=='new_reference':source_check(path,protocol)
            if bank=='holdout':assert (path/'manifest.json').stat().st_mtime>=barrier['timestamp']
            out['evaluations'][f'{bank}_{name}']=summary;out['startups'][f'{bank}_{name}']=startup;rr[name]=rows;gg[name]=g
        baselines=list(protocol['development_baselines'])
        for seed in finals:
            out['comparisons'][f'{bank}_{seed}']={b:compare(rr[seed],rr[b],gg[seed]) for b in baselines}
            out['comparisons'][f'{bank}_{seed}_strict']={b:compare([x for x in rr[seed] if x['strict_transfer']],rr[b],gg[seed]) for b in baselines}
        out['reference_repair_comparisons'][bank]={b:compare(rr['new_reference'],rr[b],gg['new_reference']) for b in baselines if b!='new_reference'}
    check(protocol);out['passed']=True;(a.output/'verification.json').write_text(json.dumps(out,indent=2));shutil.copy2(__file__,a.output/'verification_source.py')
    print(json.dumps(dict(passed=True,training=out['training'],outcomes={k:{z:v[z] for z in ['successes','physical_failures','trials']} for k,v in out['evaluations'].items()}),indent=2))
if __name__=='__main__':main()
