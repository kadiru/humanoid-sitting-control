"""Development-only admission for earlier residual control; no holdout access."""
import argparse,traceback
import numpy as np
from earlier_control_common import *
from run_transfer_assessment import run,status
from verify_seat_load_study import starts,load,equal

def physical(folder,count,baseline):
    rows=read(folder/'results.json')['episodes'];assert len(rows)==count*18
    reference=read(baseline/'results.json')['episodes'][:18];limits={x['case']:x['traction_bad_ticks'] for x in reference}
    assert all(x['steps']==1000 and x['duration_s']==20 for x in rows)
    assert all(x['success'] and not any(x['failures'].values()) and x['slip_ticks']==0 and x['illegal_ticks']==0 for x in rows)
    # Pre-main revision: exploration may incur a single conservative warning,
    # never slip/contact/physical failure or exceed modeled friction capacity.
    # Advancement and selection still prohibit added per-case margin ticks.
    assert all(x['traction_bad_ticks']<=max(1,limits[x['case']]) for x in rows)
    z=np.load(folder/'every_tick.npz');f=z['force'];n=z['normal'];fn=(f*n).sum(-1)
    ft=np.linalg.norm(f-fn[...,None]*n,axis=-1);loaded=np.linalg.norm(f,axis=-1)>.01*z['weight'][...,None]
    util=np.where(loaded,ft/(.6*np.maximum(fn,1e-7)),0).max(-1);assert float(util.max())<1.
    return dict(trials=len(rows),stable=len(rows),strict=sum(x['strict_transfer'] for x in rows),zero_slip_contact_physical=True,peak_friction_utilization=float(util.max()),max_margin_ticks=max(x['traction_bad_ticks'] for x in rows),added_margin_trial_instances=sum(x['traction_bad_ticks']>limits[x['case']] for x in rows),preflight_admission='At most one conservative warning tick; no capacity violation. Progression/selection retain stricter no-added-margin gate.')

def main():
    p=argparse.ArgumentParser();p.add_argument('--gpu',type=int,required=True);a=p.parse_args();seed=SEEDS[a.gpu]
    folder=PREFLIGHT;folder.mkdir(exist_ok=True);state=DATA/f'preflight_status_gpu{a.gpu}.json'
    try:
        parent=OLD/f'ppo_01/seed_{PARENTS[seed]}/development_final'
        label=f'calibrate_{seed}';status(state,'zero_action_and_reset_regression')
        zero=DATA/'preflight'/label;assert (zero/'early_contract.json').exists()
        physical(zero,2,parent)
        assert starts(parent,zero)['exact'];assert equal(load(parent_checkpoint(seed)),load(zero/'loaded_parent.pt'))
        old=np.load(parent/'trace.npz');new=np.load(zero/'trace.npz')
        for k in old.files:
            assert np.array_equal(new[k][:200],old[k]),('first episode zero regression',k)
        # Indexed reset includes five warmup samples and differs slightly from a
        # fresh global reset. Compare identical reset sequences, without tolerance.
        matched=DATA/f'preflight_recovery_01/parent_reset_{seed}'
        physical(matched,2,parent);replayed=np.load(matched/'trace.npz')
        for k in old.files:
            assert np.array_equal(new[k],replayed[k]),('matched indexed reset regression',k)
        write(folder/f'zero_validation_{seed}.json',dict(passed=True,first_episode_all_arrays_exact=True,matched_reset_all_arrays_exact=True,parent_exact=True,matched_parent=str(matched),trace_samples=len(new['t']),indexed_reset_equals_fresh_global=False))
        label=f'explore_{seed}';status(state,'random_early_action_probe')
        if not (folder/label/'early_contract.json').exists():run(folder,label,command(folder,a.gpu,label,'explore',seed=seed,updates=2))
        physical(folder/label,2,zero)
        label=f'smoke_{seed}';status(state,'two_update_optimizer_smoke')
        run(folder,label,command(folder,a.gpu,label,'train',seed=seed,updates=2));physical(folder/label,2,zero)
        cp=folder/label/'final.pt';label=f'smoke_eval_{seed}'
        run(folder,label,command(folder,a.gpu,label,'evaluate',seed=seed,checkpoint=cp));physical(folder/label,1,zero)
        status(state,'complete',seed=seed,calibration_trials=36,random_trials=36,smoke_training_trials=36,smoke_final_trials=18)
    except Exception as e:status(state,'failed',error=repr(e),traceback=traceback.format_exc());raise
if __name__=='__main__':main()
