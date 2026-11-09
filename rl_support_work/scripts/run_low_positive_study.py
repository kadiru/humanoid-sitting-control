"""Bounded two-seed comparison; one uninterrupted optimizer process per seed."""
import argparse,time,traceback
import numpy as np
from low_positive_common import *
from positive_slope_common_v5 import write,sha,repaired_baseline
from run_transfer_assessment import run,status
from validate_low_positive_feedback_v3 import validate
from positive_slope_validation_v4 import physical,matched
from verify_seat_load_study import equal,load

def clean(r):return r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks']

def gate(folder,baseline):
    rows=read(folder/'results.json')['episodes'];base=read(baseline/'results.json')['episodes'];assert len(rows)==len(base)
    lost=[i for i,(x,b) in enumerate(zip(rows,base)) if clean(b) and not clean(x)]
    new_failure=[i for i,(x,b) in enumerate(zip(rows,base)) if any(x['failures'].values()) and not any(b['failures'].values())]
    ids=[i for i,(x,b) in enumerate(zip(rows,base)) if clean(x) and clean(b)];effort={}
    for window,tolerance in [('metrics_whole_trial',1.01),('metrics_last5s',1.02)]:
        ratio=float(np.mean([rows[i][window]['total_squared_torque_Nm2'] for i in ids])/np.mean([base[i][window]['total_squared_torque_Nm2'] for i in ids])) if ids else float('inf')
        effort[window]=dict(ratio=ratio,within=ratio<=tolerance,common_clean=len(ids))
    return dict(passed=not lost and not new_failure and all(x['within'] for x in effort.values()),lost_baseline_clean=lost,added_physical_failure=new_failure,clean=sum(clean(x) for x in rows),physical=sum(any(x['failures'].values()) for x in rows),effort=effort)

def feedback(study,gpu,label,mode,seed,bank='training',start=10.,updates=1,cp=None):
    cal=DATA/'preflight_01'/f'seed_{seed}'/'calibration/feedback_calibration.json'
    cmd=feedback_command(study,gpu,label,mode,DATA/'banks'/bank,DATA/'learning_reference.json',kind='original',seed=seed,start=start,updates=updates,cal=cal,cp=cp)
    cmd[2]='../scripts/run_low_positive_feedback_v2.py';return cmd

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);p.add_argument('--seed',type=int,required=True);a=p.parse_args()
    root=DATA/'ppo_01';state=root/f'status_{a.seed}.json';folder=root/f'seed_{a.seed}';folder.mkdir(exist_ok=False)
    try:
        protocol=read(root/'protocol.json');assert protocol['frozen'] and a.seed in protocol['seeds']
        for f,h in protocol['hashes'].items():assert sha(f)==h,f
        status(state,'training',seed=a.seed,gpu=a.gpu,start_s=10.)
        run(folder,'train',feedback(folder,a.gpu,'train','train',a.seed,updates=16))
        initial=load(folder/'train/initial.pt');smoke=load(DATA/'preflight_01'/f'seed_{a.seed}'/'smoke/initial.pt')
        assert equal(initial,smoke),'Fresh main/smoke initialization parity'
        write(folder/'training_validation.json',validate(folder/'train'))
        baseline=DATA/'preflight_01'/f'seed_{a.seed}'/'calibration';candidates=[]
        for update in [4,8,12,16]:
            label=f'development_update_{update:02d}'
            status(state,'development_selection',seed=a.seed,update=update)
            cp=folder/'train'/f'model_{update:02d}.pt';run(folder,label,feedback(folder,a.gpu,label,'evaluate',a.seed,cp=cp))
            validation=validate(folder/label);prefix=matched(baseline,folder/label,start=10.);decision=gate(folder/label,baseline)
            row=dict(update=update,start_s=10.,checkpoint=str(cp),sha256=sha(cp),evaluation=str(folder/label),gate=decision,validation=validation,matched=prefix);candidates.append(row);write(folder/'development_candidates.json',candidates)
        passing=[c for c in candidates if c['gate']['passed']]
        rank=lambda c:(-c['gate']['clean'],c['gate']['physical'],c['gate']['effort']['metrics_whole_trial']['ratio'],c['start_s'])
        selected=min(passing,key=rank) if passing else dict(parent_fallback=True,start_s=None,checkpoint=None,reason='No development checkpoint retained baseline clean set and effort gate')
        if passing:
            label='development_prospective8';cp=Path(selected['checkpoint']);status(state,'prospective_earlier_evaluation',seed=a.seed)
            run(folder,label,feedback(folder,a.gpu,label,'evaluate',a.seed,start=8.,cp=cp))
            decision=gate(folder/label,baseline);earlier=dict(update=selected['update'],start_s=8.,checkpoint=str(cp),sha256=sha(cp),evaluation=str(folder/label),gate=decision,validation=validate(folder/label),matched=matched(baseline,folder/label,start=8.),trained_at_8s=False)
            write(folder/'prospective8.json',earlier)
            if decision['passed']:selected=earlier
        write(folder/'selection.json',dict(selected=selected,final_checkpoint=str(folder/'train/final.pt'),final_sha256=sha(folder/'train/final.pt'),training_start_s=10.,all16_updates_trained_at_10s=True,selected_at=time.time(),holdout_used=False))
        status(state,'waiting_both_final_selections',seed=a.seed)
        while not all((root/f'seed_{seed}/selection.json').exists() for seed in protocol['seeds']):time.sleep(10)
        for seed in protocol['seeds']:
            assert (root/f'seed_{seed}/train/final.pt').stat().st_mtime<time.time()
            assert not read(root/f'seed_{seed}/selection.json')['holdout_used']
        write(folder/'holdout_barrier.json',dict(timestamp=time.time(),selections={str(root/f'seed_{seed}/selection.json'):sha(root/f'seed_{seed}/selection.json') for seed in protocol['seeds']}))
        # Keep matched complete comparisons on both GPUs; no policy promotion.
        for bank in ['development_old','development_transfer','holdout']:
            status(state,'final_evaluation',seed=a.seed,bank=bank)
            for label,kind in [('reference','reference'),('parent','59117')]:
                name=f'{bank}_{label}';cmd=repaired_baseline(folder,a.gpu,name,kind,bank='development');setbank(cmd,DATA/'banks'/bank);run(folder,name,cmd)
            for old_seed in [61111,61117]:
                name=f'{bank}_current_{old_seed}';cp=OLD/'ppo_01'/f'seed_{old_seed}'/'block_03/train/final.pt'
                cmd=old_command(folder,a.gpu,name,'evaluate',seed=old_seed,start=6.,cp=cp,bank='development');setbank(cmd,DATA/'banks'/bank);run(folder,name,cmd)
            name=f'{bank}_final';run(folder,name,feedback(folder,a.gpu,name,'evaluate',a.seed,bank=bank,cp=folder/'train/final.pt'))
            if not selected.get('parent_fallback') and not (selected['update']==16 and selected['start_s']==10.):
                name=f'{bank}_selected';run(folder,name,feedback(folder,a.gpu,name,'evaluate',a.seed,bank=bank,start=selected['start_s'],cp=selected['checkpoint']))
        status(state,'complete',seed=a.seed,updates=16,active=448000,warmup=10500,total=458500,selection=selected,main_training_only_at10s=True)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise

if __name__=='__main__':main()
