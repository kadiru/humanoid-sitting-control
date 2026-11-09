"""New physical-clock and separation audit; never changes original verdicts."""
import json,sys
from pathlib import Path
import numpy as np
from recording import write

def main(out):
    z=dict(np.load(out/'substeps.npz',allow_pickle=False))
    d=dict(np.load(out/'trajectory.npz',allow_pickle=False))
    signals=json.loads((out/'controller_signals.json').read_text())
    assert len(z['physics_step'])==4000 and np.all(np.diff(z['physics_step'])==1)
    assert all(np.isfinite(v).all() for v in z.values())
    mass=z['mass'][:,0,:];v=z['velocity'][:,0,:,:3]
    momentum=(mass[:,:,None]*v).sum(1)
    # Compare strictly matching3999 five-millisecond intervals, never force*20ms.
    force=z['normal'].sum(1)+z['friction'].sum(1)
    gravity=np.c_[np.zeros(len(mass)),np.zeros(len(mass)),-9.81*mass.sum(1)]
    impulse=(force[1:]+gravity[1:])*.005
    residual=impulse-np.diff(momentum,axis=0)
    quiet=z['velocity'][-1000:,0]
    gap_candidate=np.array([s['separation_shadow']['candidate'] for s in signals])
    # signals and geometry are pre-step; physical support[t-1] is aligned label.
    declared=gap_candidate[1:];labels=d['support_bw'][:-1,0]
    r=dict(record_integrity_passed=True,physics_samples=4000,substep_dt_s=.005,
           actual_mass_kg=float(mass[-1].sum()),
           final5s_mean_reported_net_force_N=(force[-1000:]+gravity[-1000:]).mean(0).tolist(),
           final5s_reported_impulse_Ns=((force[-1000:]+gravity[-1000:])*.005).sum(0).tolist(),
           final5s_direct_momentum_change_Ns=(momentum[-1]-momentum[-1001]).tolist(),
           final5s_impulse_residual_Ns=residual[-1000:].sum(0).tolist(),
           maximum_direct_link_speed_m_s=float(np.max(np.linalg.norm(quiet[:,:,:3],axis=-1))),
           mean_unfiltered_minus_filtered_normal_N=(z['all_body_normal'].reshape(4000,-1,3).sum(1)-z['normal'].sum(1))[-1000:].mean(0).tolist(),
           separation_candidates=int(declared.sum()),false_separation_against_reported_load=int((declared&(labels>=.01)).sum()),
           absent_load_samples=int((labels<.01).sum()),
           declared_absent_coverage=float((declared&(labels<.01)).sum()/max(1,(labels<.01).sum())),
           source_scope='Direct simulator readouts are audit-only; momentum completeness and motor interpretation still require review.',
           force_estimator_used_by_controller=False,separation_used_by_controller=False,
           complete_rest_and_return_cycle=False,PPO_updates=0)
    write(out/'substep_audit.json',r);print(json.dumps(r))

if __name__=='__main__':main(Path(sys.argv[1]))
