"""Post-hoc functional audit; no rollout changes and no checkpoint selection."""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
from humanoid_sitting.support_split_observation import install_split_linear

p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True)
a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(exist_ok=False)
torch.set_num_threads(1)
initial=torch.load(run/'initial.pt',map_location='cpu',weights_only=False)
final=torch.load(run/'final.pt',map_location='cpu',weights_only=False)
with np.load(run/'controller.npz') as f:x=torch.from_numpy(f['policy']).flatten(0,1)
obs=TensorDict({'policy':torch.zeros(16,151)},[16])
model=MLPModel(obs,{'actor':['policy']},'actor',23,hidden_dims=[128,128],obs_normalization=True,
    distribution_cfg=dict(class_name='rsl_rl.modules.distribution:GaussianDistribution',init_std=.04,std_type='log'))
install_split_linear(model);model.eval()
report={}
for name,state in [('initial',initial),('trained',final)]:
    model.load_state_dict(state['actor_state_dict']);deltas=[];patch_missing=[]
    with torch.no_grad():
        for b in x.split(512):
            valid=b[:,148]>0
            if not valid.any():continue
            b=b[valid];changed=b.clone();changed[:,144]+=.01
            old=model(TensorDict({'policy':b},[len(b)]))
            new=model(TensorDict({'policy':changed},[len(b)]))
            difference=(.08*(old.clamp(-1,1)-new.clamp(-1,1))).abs().max(1).values.numpy()
            deltas.extend(difference.tolist())
            patch_missing.extend(difference[(b[:,140]==0).numpy()].tolist())
    report[name]=dict(valid_samples=len(deltas),missing_patch_samples=len(patch_missing),
        clearance_plus_1cm_max_residual_target_change_rad=float(max(deltas,default=0)),
        clearance_plus_1cm_missing_patch_max_residual_target_change_rad=float(max(patch_missing,default=0)),
        new_column_max_weights=state['actor_state_dict']['mlp.0.weight'][:,144:].abs().max(0).values.tolist())
normalization_unchanged=all(torch.equal(v,final[role][k]) for role in ('actor_state_dict','critic_state_dict')
    for k,v in initial[role].items() if k.startswith('obs_normalizer.'))
accounting=json.loads((run/'attempt_accounting.json').read_text())
verification=json.loads((run/'verification.json').read_text())
total=accounting['completed_transitions']+accounting['unfinished_transitions']
report.update(normalization_bitwise_unchanged=normalization_unchanged,accounted_transitions=total,
    declared_transitions=verification['transitions'],
    note='Counterfactual scalar-only perturbation on recorded training observations. Does not establish a causal success improvement; final motor slew limits are not modeled here.')
assert total==verification['transitions']==32768 and normalization_unchanged
assert report['initial']['clearance_plus_1cm_max_residual_target_change_rad']==0
assert report['trained']['clearance_plus_1cm_missing_patch_max_residual_target_change_rad']>0
(out/'results.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
