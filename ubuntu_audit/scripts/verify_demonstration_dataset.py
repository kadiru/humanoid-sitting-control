"""Verify alignment and 5 ms contact histories; write small timing/index sidecars."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('datasets',nargs='+');a=p.parse_args()
for path in a.datasets:
    root=Path(path);r=json.loads((root/'results.json').read_text());z=np.load(root/'rollouts.npz');T,N,D=z['student_observations'].shape
    assert D==99 and N==r['num_trials'];assert z['targets'].shape==(T,N,37);actor_key='student_actor_observations' if 'expert_targets' in z.files else 'teacher_actor_observations'
    assert z[actor_key].shape==(T,N,99 if 'expert_targets' in z.files else 129)
    if 'expert_targets' not in z.files:assert np.max(np.abs(z['actions']))==0
    seat=z['seat_contact_history'];peak=np.linalg.norm(seat.sum(axis=3),axis=-1).max(axis=2)
    # histories T,N,4,bodies,XYZ: newest first.
    np.testing.assert_allclose(peak,z['seat_force_peak'],rtol=1e-5,atol=1e-4)
    np.testing.assert_allclose(z['all_contact_history'][:,:,0],z['body_forces'],rtol=1e-5,atol=1e-4)
    for field in z.files:assert np.isfinite(z[field]).all(),field
    pre=np.arange(T)*.02;post=pre+.02;history=post[:,None]-np.arange(4)[None,:]*.005
    np.savez_compressed(root/'synchronization.npz',pre_time_s=pre,post_time_s=post,contact_history_time_s=history,geometry_assignment=np.array(r['geometry_assignment']),success=np.array([o['success'] for o in r['outcomes']]),joint_names=np.array(r['joint_names']),contact_body_names=np.array(r['body_names']))
    metadata=dict(trajectories=N,transitions=T*N,successful=sum(o['success'] for o in r['outcomes']),geometry_seed=r['geometry_seed'],physics_samples_per_control=4,verified_contact_history_peaks=True,numeric_arrays_only=True,normal_contact_forces_only='support_friction_history' not in z.files,schema='demonstration_manifest.json and synchronization.npz',files={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ['rollouts.npz','initial_conditions.npz','geometry_bank.json','results.json','synchronization.npz']})
    if 'support_friction_history' in z.files:metadata['maximum_filtered_support_friction_N']=float(np.linalg.norm(z['support_friction_history'],axis=-1).max())
    (root/'verification.json').write_text(json.dumps(metadata,indent=2)+'\n');print(path,metadata['trajectories'],metadata['transitions'],metadata['successful'])
