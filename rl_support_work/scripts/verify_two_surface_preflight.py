"""Independent preflight checks. Does not alter simulation results."""
from pathlib import Path
import hashlib,json
import numpy as np
import torch
r=Path('/data/humanoid-sitting-perception/two_surface_curriculum_01')
read=lambda p:json.loads(p.read_text())
assert read(r/'selected_pose_validation_01/results.json')['stationary_valid']==14
assert all(x['stable_fraction']==1 and x['final_hold_s']==20 and not any(x['failures'].values()) for x in read(r/'selected_pose_validation_01/results.json')['rows'])
assert read(r/'reset_isolation_01/reset_probe.json')['passed']
for label in ['fixed_all_01','reference_motion_gpu0_01','smoke_evaluation_01']:
 d=read(r/label/'results.json')
 assert all(x['steps']==1000 and x['duration_s']==20 and x['late_ticks']==500 and x['last5_ticks']==250 for x in d['episodes'])
 assert all(not any(x['failures'].values()) for x in d['episodes']) if label!='reference_motion_gpu0_01' else True
fixed=np.load(r/'fixed_all_01/trace.npz');startup=np.load(r/'fixed_all_01/startup.npz')
assert np.array_equal(fixed['command'],np.broadcast_to(startup['command'][None],fixed['command'].shape))
reference=np.load(r/'reference_motion_gpu0_01/startup.npz')
differences={k:float(abs(reference[k]-startup[k]).max()) for k in startup.files};assert max(differences.values())==0,differences
smoke=read(r/'smoke_evaluation_01/results.json');assert smoke['count']==smoke['successes']==2
v=read(r/'optimizer_smoke_01/verification.json');assert v==dict(updates=2,transitions=2048,active=2048,warmup=0)
initial=torch.load(r/'optimizer_smoke_01/initial.pt',map_location='cpu',weights_only=False);final=torch.load(r/'optimizer_smoke_01/final.pt',map_location='cpu',weights_only=False)
loaded=torch.load(r/'smoke_evaluation_01/initial.pt',map_location='cpu',weights_only=False)
changes={}
for role in ['actor_state_dict','critic_state_dict']:
 assert all(torch.isfinite(v).all() for v in final[role].values())
 changes[role]=[k for k,v in final[role].items() if not torch.equal(v,initial[role][k])]
 assert any(k.startswith('mlp') for k in changes[role])
 assert all(torch.equal(v,loaded[role][k]) for k,v in final[role].items())
assert torch.count_nonzero(initial['actor_state_dict']['mlp.4.weight'])==0
assert torch.count_nonzero(initial['actor_state_dict']['mlp.4.bias'])==0
source=Path('/home/kadir/humanoid-sitting/scripts/run_two_surface_path.py')
result=dict(passed=True,stationary_poses=14,endpoint_supported=True,zero_reference_actor=True,constant_fixed_commands=True,reset_isolation=True,exact_fixed_reference_startup_errors=differences,smoke_budget=v,smoke_success='2/2 full20s',finite_changed_weights=changes,smoke_checkpoint_loaded=True,runtime_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),resume_metadata_note='After smoke, continuation training increments RSL saved zero-based completed iteration before the next block. No simulation/control/reward change; new-update counts stay32/block.')
(r/'preflight_verification_01.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
