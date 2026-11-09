"""Record evidence that the authorized pilot reached real optimizer updates."""
import datetime
import hashlib
import json
from pathlib import Path
import torch

root=Path('/data/humanoid-sitting-perception/resting_bounded_ppo_02')
def read(p):return json.loads(p.read_text())
manifest=read(root/'frozen/manifest.json')
assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==sha for p,sha in manifest.items())
baselines={}
for bank,n in [('starts',16),('heights',12),('impulse',12),('novel',16)]:
    d=read(root/f'fixed_{bank}/results.json')
    assert d['successes']==n
    assert all(r['steps']==1000 and r['duration_s']==20 and not any(r['failures'].values()) for r in d['episodes'])
    baselines[bank]=dict(success=n,total=n)
train=root/'train_53111';progress=read(train/'progress.json')
assert progress['update']>0 and progress['transitions']==progress['update']*1024
assert progress['active_transitions']+progress['warmup_transitions']==progress['transitions']
assert read(train/'initial_policy_check.json')['zero_mean_joint_increment']
initial=torch.load(train/'initial.pt',map_location='cpu',weights_only=False)
first=torch.load(train/'model_0.pt',map_location='cpu',weights_only=False)
changes={role:max(float(abs(value-first[role][key]).max()) for key,value in initial[role].items())
         for role in ('actor_state_dict','critic_state_dict')}
assert all(value>0 for value in changes.values())
assert all(torch.isfinite(value).all() for role in changes for value in first[role].values())
report=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),baselines=baselines,
    frozen_sources_verified=len(manifest),initial_mean_zero=True,
    initial_to_first_saved_update_max_parameter_changes=changes,finite=True,progress=progress,
    checkpoint_sha256=hashlib.sha256((train/'model_0.pt').read_bytes()).hexdigest(),
    final_policy_quality_not_yet_known=True)
out=root/'first_optimizer_validation.json';assert not out.exists()
out.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
