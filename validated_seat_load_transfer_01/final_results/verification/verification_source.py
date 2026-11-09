"""Independent final-checkpoint, budget, startup and paired-outcome audit."""
import argparse,json,hashlib,shutil,time
from pathlib import Path
import numpy as np
import torch
from run_seat_load_study import read,sha,check,DATA,OLD

WINDOWS=['metrics_whole_trial','metrics_last10s','metrics_last5s']
METRICS=['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2','vertical_support_load_bw','foot_load_bw','max_friction_utilization']

def equal(a,b):
    if torch.is_tensor(a):return torch.equal(a,b)
    if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
    return a==b

def load(p):return torch.load(p,map_location='cpu',weights_only=False)

def evaluation(path,count,expected_checkpoint=None):
    data=read(path/'results.json');rows=data['episodes'];g=read(path/'geometry.json');trace=np.load(path/'trace.npz');start=np.load(path/'startup.npz')
    assert len(rows)==count and {x['case'] for x in rows}==set(range(count))
    assert all(x['steps']==1000 and x['duration_s']==20 and x['late_ticks']==500 and x['last5_ticks']==250 for x in rows)
    assert len(trace['t'])==200 and abs(trace['t'][-1]-20)<1e-8 and abs(trace['t'][0]-.1)<1e-8
    assert all(x['success']==(not any(x['failures'].values()) and x['final_hold_s']>=5) for x in rows)
    assert np.isfinite(trace['command']).all() and np.isfinite(trace['torque']).all()
    names=data['joint_names'];fingers=[i for i,n in enumerate(names) if any(z in n for z in ['_zero_','_one_','_two_','_three_','_four_','_five_','_six_'])]
    assert len(fingers)==14 and np.all(trace['command'][:,:,fingers]==start['command'][None,:,fingers])
    sequence=np.concatenate((start['command'][None],trace['command']),axis=0)
    slew=float(abs(np.diff(sequence,axis=0)).max());assert slew<=.015001,slew
    manifest=read(path/'manifest.json')
    if expected_checkpoint:
        assert Path(manifest['args']['checkpoint'])==expected_checkpoint
        checkpoint=load(expected_checkpoint);loaded=load(path/'initial.pt')
        for role in ['actor_state_dict','critic_state_dict']:assert equal(checkpoint[role],loaded[role])
    success=[x['case'] for x in rows if x['success']];physical=[x['case'] for x in rows if any(x['failures'].values())]
    # Independently cross-check the all-joint identity on the sampled trace.
    error=float(abs((trace['torque']**2).sum(-1)-trace['effort']).max());assert error<.002,error
    audit=read(path/'sensor_audit.json')['truth_validation'];failtimes={x['case']:x['t'] for x in read(path/'first_failures.json')}
    relevant=[x for x in audit if x['t']<=failtimes.get(x['case'],20)]
    valid=[x for x in relevant if x['valid']]
    geo=dict(observations=len(relevant),unavailable=sum(not x['valid'] for x in relevant),max_offset_error_m=max(abs(x['offset_error_m']) for x in valid),max_normal_error_deg=max(x['normal_error_deg'] for x in valid))
    before=np.stack([trace['t']<=failtimes.get(i,20) for i in range(count)],axis=1)
    if 'estimated_root_x' in trace:
        pelvis=data['body_names'].index('pelvis');truth=trace['body'][:,:,pelvis,0]-np.array(data['origins'])[None,:,0]
        error_x=trace['estimated_root_x']-truth
        geo['foot_anchored_root_x_rms_before_failure_m']=float(np.sqrt((error_x[before]**2).mean()))
        geo['foot_anchored_root_x_max_before_failure_m']=float(abs(error_x[before]).max())
    feet=[data['body_names'].index(s+'_ankle_roll_link') for s in ['left','right']]
    move=np.linalg.norm(trace['body'][:,:,feet,:2]-trace['body'][0:1,:,feet,:2],axis=-1).max(-1)
    geo['max_foot_horizontal_displacement_since_0p1s_before_failure_m']=float(move[before].max())
    summary=dict(successes=len(success),trials=count,physical_failures=len(physical),success_cases=success,physical_failure_cases=physical,failure_labels={k:sum(x['failures'][k] for x in rows) for k in rows[0]['failures']},sampled_command_slew_max_rad=slew,fixed_fingers_exact=True,torque_identity_max_error=error,geometry_before_failure=geo)
    summary['by_surface']=[dict(case=x['case'],height_m=g[x['geometry_index']]['height'],slope_deg=g[x['geometry_index']]['pitch_deg'],success=x['success'],failures=x['failures'],final_hold_s=x['final_hold_s']) for x in rows]
    return summary,rows,g

def starts(reference,path):
    a=np.load(reference/'startup.npz');b=np.load(path/'startup.npz');result={}
    for k in ['root','q','command','geometry','observation']:
        assert a[k].shape==b[k].shape,(k,a[k].shape,b[k].shape)
        result[k]=float(abs(a[k]-b[k]).max())
    # Different physical GPUs may have small floating-point startup differences.
    # Never silently loosen this bound; a failure requires inspection.
    assert max(result.values())<=1e-5,result
    ga=read(reference/'geometry.json');gb=read(path/'geometry.json')
    for x,y in zip(ga,gb):
        for k in ['height','pitch_deg','top_center','normal','depth','width','rotation_xyzw']:assert x[k]==y[k],k
    return dict(max_errors=result,exact=max(result.values())==0,tolerance=1e-5)

def compare(rows,reference,geometry):
    ref={x['case']:x for x in reference};common=[x for x in rows if x['success'] and ref[x['case']]['success']]
    groups={'all':common}
    for x in common:
        g=geometry[x['geometry_index']]
        for name in [f"height_{g['height']:.4f}",f"slope_{g['pitch_deg']:+.2f}"]:
            groups.setdefault(name,[]).append(x)
    output={}
    for name,items in groups.items():
        output[name]={}
        for window in WINDOWS:
            output[name][window]={}
            for metric in METRICS:
                a=np.array([x[window][metric] for x in items]);b=np.array([ref[x['case']][window][metric] for x in items]);am=float(a.mean()) if len(a) else None;bm=float(b.mean()) if len(b) else None
                output[name][window][metric]=dict(common_successes=len(items),policy=am,baseline=bm,change_percent=100*(am/bm-1) if bm and am is not None else None)
    return dict(common_success_cases=[x['case'] for x in common],grouped=output)

def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    protocol=read(a.study/'protocol.json');check(protocol);barrier=read(a.study/'holdout_barrier.json')
    result=dict(passed=False,source_hashes=len(protocol['hashes']),prior_hashes=len(protocol['prior_hashes']),training={},evaluations={},startup={},comparisons={},strong_rest_goal={},caveats=['Reference-assisted residual PPO; two seeds and repeated finite rigid surfaces.','Rest-only targets and standing transitions are separate outcomes.','No savings from failed fragments; torque is not electrical energy or a global optimum.','All-tick physical limits come from frozen runtime code and episode flags; command-slew and finger checks also use100ms traces.'])
    for seed in protocol['seeds']:
        folder=a.study/f'train_{seed}';v=read(folder/'verification.json');initial=load(folder/'initial.pt');warm=load(folder/'critic_warmup.pt');final=load(folder/'final.pt')
        assert v['updates']==64 and v['actor_updates']==56 and v['transitions']==184320 and v['policy_decisions']==36864
        assert v['active']+v['warmup']==v['transitions']
        assert equal(initial['actor_state_dict'],warm['actor_state_dict'])
        assert final['iter']==63
        for role in ['actor_state_dict','critic_state_dict']:
            assert all(torch.isfinite(x).all() for x in final[role].values());assert any(not torch.equal(x,final[role][k]) for k,x in initial[role].items())
        assert initial['actor_state_dict']['mlp.0.weight'].shape==(64,406)
        assert initial['actor_state_dict']['mlp.4.weight'].shape==(23,64)
        assert initial['actor_state_dict']['mlp.4.weight'].count_nonzero()==0 and initial['actor_state_dict']['mlp.4.bias'].count_nonzero()==0
        opt=final['optimizer_state_dict'];steps=[int(x['step']) for x in opt['state'].values()];assert steps and min(steps)>0 and max(steps)<=1024
        assert set(warm['optimizer_state_dict']['state'])==set(range(7,13))
        assert all(int(opt['state'][i]['step'])==896 for i in range(7))
        assert all(int(opt['state'][i]['step'])==1024 for i in range(7,13))
        assert opt['param_groups'][0]['lr']==1e-4
        episodes=read(folder/'results.json')['episodes'];pending=25*len(episodes)-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in episodes)
        assert 0<=pending<=450 and 0<=unfinished<18000
        result['training'][str(seed)]=dict(**v,pending_warmup_ticks=pending,unfinished_active_ticks=unfinished,initial_settle_ticks_excluded=450,optimizer_step_counts=steps)
        frozen=barrier['checkpoints'][str(seed)];assert frozen['sha256']==sha(folder/'final.pt') and frozen['time']<=barrier['time']
    for bank in ['development','holdout']:
        paths={k:Path(v) for k,v in protocol['development_baselines'].items()} if bank=='development' else {k:a.study/('holdout_'+k) for k in ['current_reference','current_56211','current_56217','new_reference']}
        for seed in protocol['seeds']:paths[str(seed)]=a.study/f'{bank}_{seed}'
        rows={};geoms={}
        for label,path in paths.items():
            cp=a.study/f'train_{label}/final.pt' if label.isdigit() else None
            if label in ['current_56211','current_56217']:
                cp=OLD/f"sitting_ppo_01/train_{label.split('_')[1]}/final.pt"
                parent=read(path/'parent_identity.json');assert parent['sha256']==sha(parent['path'])
            summary,rr,gg=evaluation(path,18 if bank=='development' else 16,cp);rows[label]=rr;geoms[label]=gg
            result['evaluations'][f'{bank}_{label}']=summary;result['startup'][f'{bank}_{label}']=starts(paths['current_reference'],path)
            if bank=='holdout':assert (path/'manifest.json').stat().st_mtime>=barrier['time']
        for seed in protocol['seeds']:
            label=str(seed);result['comparisons'][f'{bank}_{seed}']={b:compare(rows[label],rows[b],geoms[label]) for b in ['current_reference','current_56211','current_56217','new_reference']}
            reference={x['case']:x for x in rows['current_reference']};qualified=[]
            for x in rows[label]:
                m=x['metrics_last5s'];b=reference[x['case']]['metrics_last5s']
                if x['success'] and m['vertical_support_load_bw']>=.5 and m['total_squared_torque_Nm2']<=.35*b['total_squared_torque_Nm2']:qualified.append(x['case'])
            result['strong_rest_goal'][f'{bank}_{seed}']=dict(count=len(qualified),cases=qualified,definition='Full stable success;final5s mean seat vertical load>=.5BW and torque squared<=35% of matched current reference. Baseline torque is descriptive, not a claim of savings from a failed reference.')
    check(protocol);result['passed']=True
    (a.output/'verification.json').write_text(json.dumps(result,indent=2));shutil.copy2(__file__,a.output/'verification_source.py')
    print(json.dumps({k:v for k,v in result.items() if k not in ['comparisons','evaluations','startup']},indent=2))

if __name__=='__main__':main()
