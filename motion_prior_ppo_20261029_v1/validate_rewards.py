"""Independent saved-trajectory ranking and telescoping tests; CPU only."""
import argparse
from pathlib import Path
import numpy as np
import torch
from common import ROOT,read,write,sha
from rewards import staged_reward
from audit import reward_numpy

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args()
    torch.set_num_threads(1)
    roots=[ROOT/'outputs/full_body_contact_imitation_v4/teacher_replay_sitting',
           ROOT/'outputs/recovery_imitation_sitting_20261028_v2/clone_evaluation_64111',
           ROOT/'outputs/recovery_imitation_sitting_20261028_v3/eval_sitting_64111_128',
           ROOT/'outputs/recovery_imitation_sitting_20261028_v3/eval_sitting_64111_256']
    results=[];all_failed=[];all_clean=[];discount=.999**np.arange(1000)
    for root in roots:
        audit=read(root/'independent_audit.json');assert audit['passed']
        with np.load(root/'trajectory.npz',allow_pickle=False) as z:
            names=['alive','legal','good','phase','q_error','speed','gyro','support_bw','held_ticks','loaded_slip','wrong_force','peak_bw','normalized_command_delta','first_failure','clean_completion','stable_completion','torque_squared','learn']
            d={k:z[k] for k in names}
        target=reward_numpy(d)*d['learn']
        previous=torch.zeros(72);actual=[]
        for tick in range(1000):
            x={k:torch.from_numpy(v[tick]) for k,v in d.items()}
            x.update(previous_potential=previous,horizon=torch.full((72,),tick==999))
            r,terms=staged_reward(x);previous=terms['potential'];actual.append(r.numpy()*d['learn'][tick])
        actual=np.stack(actual)
        np.testing.assert_allclose(actual,target,rtol=3e-6,atol=3e-6)
        returns=(target*discount[:,None]).sum(0)
        failures=d['first_failure'].any(0);first=d['first_failure'].argmax(0)
        n=first[failures]+1
        expected=.02*(1-.999**n)/(1-.999)-20*.999**(n-1)
        np.testing.assert_allclose(returns[failures],expected,rtol=3e-6,atol=3e-6)
        all_failed.extend(zip(n.tolist(),returns[failures].tolist()))
        all_clean.extend(returns[np.array(audit['clean'])].tolist())
        results.append(dict(source=str(root),trajectory_sha256=sha(root/'trajectory.npz'),clean_count=audit['clean_count'],
            mean_discounted_return=float(returns.mean()),returns=returns.tolist(),first_failure_controls=np.where(failures,first+1,1000).tolist(),
            reward_implementation_max_difference=float(abs(actual-target).max())))
    assert all_clean and min(all_clean)>max(x[1] for x in all_failed)
    ordered=sorted(all_failed)
    worst=0.
    for (n,a),(m,b) in zip(ordered,ordered[1:]):
        if m>n:assert b>a,(n,m,a,b)
        else:worst=max(worst,abs(b-a))
    write(Path(args.output),
        dict(passed=True,discount=.999,all_completed_original_clean_trials_outrank_failures=True,
        earlier_failed_trial_has_lower_discounted_return=True,equal_failure_time_max_score_difference=worst,
        preserved_physical_and_action_thresholds=True,results=results))

if __name__=='__main__':main()
