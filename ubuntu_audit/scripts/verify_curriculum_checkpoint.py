"""Verify actual parameter updates and finite weights in a local curriculum run."""
import argparse,json
from pathlib import Path
import torch
p=argparse.ArgumentParser();p.add_argument('run');p.add_argument('--reference');args=p.parse_args()
root=Path(args.run);models=sorted(root.glob('model_*.pt'),key=lambda p:int(p.stem.split('_')[-1]))
first=Path(args.reference) if args.reference else models[0];last=models[-1]
a=torch.load(first,map_location='cpu',weights_only=False);b=torch.load(last,map_location='cpu',weights_only=False);changes={}
for group,items in b.items():
    if not isinstance(items,dict):continue
    deltas=[]
    for name,value in items.items():
        if not torch.is_tensor(value):continue
        assert torch.isfinite(value).all(),(group,name)
        old=a.get(group,{}).get(name)
        if torch.is_tensor(old) and value.is_floating_point():deltas.append((value-old).abs().max().item())
    if deltas:changes[group]=max(deltas)
assert changes.get('actor_state_dict',0)>0 and changes.get('critic_state_dict',0)>0,changes
result=dict(checkpoint=str(last.resolve()),reference=str(first.resolve()),iteration=b.get('iter'),bytes=last.stat().st_size,max_parameter_changes=changes)
(root/'checkpoint_verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
