"""Independent completion checks and paired full-duration effort comparisons."""
import argparse,json,shutil
from pathlib import Path
import numpy as np
from verify_split_leaning import evaluation,starts,read,sha,load,equal,weights,check_manifest
from run_transfer_effort_study import retention,DATA

WINDOWS=['metrics_whole_trial','metrics_last10s','metrics_last5s']
METRICS=['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']
def compare(rows,base,geoms):
    pairs=[]
    for a,b in zip(rows,base):
        assert (a['case'],a['geometry_index'],a['start_stage'])==(b['case'],b['geometry_index'],b['start_stage'])
        if not(a['success'] and b['success']):continue
        g=geoms[a['geometry_index']]
        for window in WINDOWS:
            for metric in METRICS:
                v=a[window][metric];ref=b[window][metric]
                pairs.append(dict(case=a['case'],label=g['label'],height_m=g['height'],slope_deg=g['pitch_deg'],known_anchor=g['known_anchor'],window=window,metric=metric,policy=v,baseline=ref,change_percent=100*(v/ref-1) if ref else None))
    grouped={}
    for window in WINDOWS:
        grouped[window]={}
        for metric in METRICS:
            use=[p for p in pairs if p['window']==window and p['metric']==metric]
            pv=float(np.mean([p['policy'] for p in use])) if use else None;bv=float(np.mean([p['baseline'] for p in use])) if use else None
            grouped[window][metric]=dict(common_successes=len(use),policy=pv,baseline=bv,change_percent=100*(pv/bv-1) if bv else None)
    return dict(grouped=grouped,pairs=pairs)

def verify(folder):
    p=read(folder/'protocol.json');surface=p['surface'];assessment=DATA/'assessment'/surface
    report=dict(passed=False,surface=surface,training={},evaluations={},starts={},selection={},comparisons={},by_geometry={})
    for i,(path,h) in enumerate(p['hashes'].items()):assert sha(path)==h==sha(folder/'frozen'/f'{i:03d}_{Path(path).name}'),path
    for path,h in p['prior_hashes'].items():assert sha(path)==h,path
    report.update(frozen_hashes=len(p['hashes']),prior_hashes=len(p['prior_hashes']))
    ref=read(assessment/'reference/results.json')['episodes'];datasets={};locations={}
    for seed,oldseed in zip(p['seeds'],p['parent_seeds']):
        state=read(folder/f'seed_{seed}.json');prev=Path(state['parent']);parent=read(assessment/f'policy_{oldseed}/results.json')['episodes'];total={k:0 for k in ['updates','transitions','active','warmup']};history=[];eligible=[]
        assert len(state['history'])==4
        for block,row in enumerate(state['history']):
            train=folder/row['train'];m=check_manifest(train);v=read(train/'verification.json');initial=load(train/'initial.pt');final=load(train/'final.pt');old=load(prev)
            assert all(equal(initial[k],old[k]) for k in ['actor_state_dict','critic_state_dict','optimizer_state_dict'])
            assert initial['iter']==old['iter']+1 and final['iter']==initial['iter']+31
            assert m['args']['start_stage']=='0' and m['args']['objective']=='effort' and m['args']['seed']==seed
            assert sha(m['args']['calibration'])==sha(train/'reward_calibration.json')
            assert v['updates']==32 and v['transitions']==36864 and v['active']+v['warmup']==36864
            changed=weights(initial,final);episodes=read(train/'results.json')['episodes'];pending=25*len(episodes)-v['warmup'];unfinished=v['active']-sum(x['steps'] for x in episodes)
            assert 0<=pending<=450 and 0<=unfinished<=18000
            for k in total:total[k]+=v[k]
            prev=train/'final.pt';label=row['evaluation'];ev,data=evaluation(folder/label,18,prev);report['evaluations'][label]=ev;report['starts'][label]=starts(assessment/'reference',folder/label)
            gate=retention(data['episodes'],ref,parent);assert gate==row['retention']
            if gate['eligible']:eligible.append((block,gate,prev))
            datasets[label]=data['episodes'];locations[label]=folder/label
            history.append(dict(block=block,budget=v,changed=changed,pending_warmup=pending,unfinished_active=unfinished,checkpoint_sha256=sha(prev)))
        assert total['updates']==128 and total['transitions']==147456
        assert state['final_checkpoint']==str(prev)
        best=min(eligible,key=lambda x:(-x[1]['successes'],x[1]['mean_squared_torque'])) if eligible else None
        assert state['selected_checkpoint']==(str(best[2]) if best else state['parent'])
        report['selection'][str(seed)]=dict(selected_checkpoint=state['selected_checkpoint'],selected_block=best[0] if best else None,parent_fallback=best is None,selection_used_holdout=False)
        report['training'][str(seed)]=dict(**total,blocks=history)
    jobs=[('holdout_reference',None)]
    for seed,oldseed in zip(p['seeds'],p['parent_seeds']):
        state=read(folder/f'seed_{seed}.json');jobs.extend([(f'holdout_parent_{oldseed}',state['parent']),(f'holdout_final_{seed}',state['final_checkpoint'])])
        if state['selected_checkpoint'] not in [state['final_checkpoint'],state['parent']]:jobs.append((f'holdout_selected_{seed}',state['selected_checkpoint']))
    for label,cp in jobs:
        ev,data=evaluation(folder/label,16,Path(cp) if cp else None);report['evaluations'][label]=ev;report['starts'][label]=starts(folder/'holdout_reference',folder/label)
        datasets[label]=data['episodes'];locations[label]=folder/label
    for label,rows in datasets.items():
        geoms=read(locations[label]/'geometry.json');report['by_geometry'][label]=[]
        for j,g in enumerate(geoms):
            use=[e for e in rows if e['geometry_index']==j]
            report['by_geometry'][label].append(dict(label=g['label'],height_m=g['height'],slope_deg=g['pitch_deg'],count=len(use),success=sum(x['success'] for x in use),physical_failures=sum(any(x['failures'].values()) for x in use)))
    for seed,oldseed in zip(p['seeds'],p['parent_seeds']):
        state=read(folder/f'seed_{seed}.json');final=f'dev_{seed}_block3';selected=f'dev_{seed}_block{state["selected_block"]}' if state['selected_block'] is not None else None
        for label in set([final]+([selected] if selected else [])):
            report['comparisons'][label]={'corrected_reference':compare(datasets[label],ref,read(locations[label]/'geometry.json')),'parent':compare(datasets[label],read(assessment/f'policy_{oldseed}/results.json')['episodes'],read(locations[label]/'geometry.json'))}
        labels=[f'holdout_final_{seed}']
        if f'holdout_selected_{seed}' in datasets:labels.append(f'holdout_selected_{seed}')
        for label in labels:
            report['comparisons'][label]={'corrected_reference':compare(datasets[label],datasets['holdout_reference'],read(locations[label]/'geometry.json')),'parent':compare(datasets[label],datasets[f'holdout_parent_{oldseed}'],read(locations[label]/'geometry.json'))}
    report['passed']=True;return report
def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    shutil.copy2(__file__,a.output/'verification_source.py');v=verify(a.data);(a.output/'verification.json').write_text(json.dumps(v,indent=2,allow_nan=False))
    print(json.dumps({'passed':v['passed'],'surface':v['surface'],'evaluations':{k:dict(success=x['successes'],failures=x['physical_failures']) for k,x in v['evaluations'].items()}},indent=2))
if __name__=='__main__':main()
