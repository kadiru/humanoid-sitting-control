"""CPU-only saved-data follow-up: case conditioning and housing reward signal."""
import os
os.environ['CUDA_VISIBLE_DEVICES']=''
import json,hashlib,time
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(1)
P=Path('/data/humanoid-sitting-perception/low_seat_contact_01/contact_reward_comparison_128_v4')
out={'sources':{},'policy_variation':{},'housing_signal':{}}
def bind(p):
    out['sources'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest();return p
def read(p):return json.loads(bind(p).read_text())
for arm,seed,stage in [('original',63111,32),('original',63117,32),('revised',63111,32),('revised',63117,32),('revised',63111,64)]:
    targets=[]
    for bank,ids in [('known_old',[4,5,6,7]),('known_transfer',[4,5]),('known_recent',[4,5])]:
        p=P/f'eval_{arm}_{seed}_{stage:03d}_{bank}'
        assert read(p/'independent_audit.json')['passed']
        b=torch.load(bind(p/'decision_batches.pt'),map_location='cpu',weights_only=False)[0]
        ds=np.array(b['decision_steps']);t=.004*np.tanh(b['old_mean'].numpy())
        targets.append(t[:,ids,:])
    t=np.concatenate(targets,axis=1);v=t[(ds>=90)&(ds<130)]
    spread=np.ptp(v,axis=1)
    out['policy_variation'][f'{arm}_{seed}_{stage}']={'landing_case_spread_max_rad':float(spread.max()),'landing_case_spread_mean_rad':float(spread.mean()),'landing_command_abs_max_rad':float(np.abs(v).max()),'scope':'Same-time target variation across eight failed case identities. Small variation does not prove observations lack information.'}
for seed,stages in [(63111,[16,32,64]),(63117,[16,32])]:
    records=[]
    for stage in stages:
        p=P/f'train_revised_{seed}_{stage:03d}'
        audit=read(p/'reward_audit.json');contract=read(p/'feedback_contract.json')
        with np.load(bind(p/'reward_geometry.npz')) as z:g=dict(z)
        for it,a in enumerate(audit):
            for j in [2,3]:
                c=g['housing_vertex_clearance_m'][it,:,j]
                records.append({'update':contract['initial_updates']+it+1,'training_case':j,'clean':a['clean'][j],
                    'event_totals_raw':{k:v[j] for k,v in a['event_totals_raw'].items()},
                    'component_totals_raw':{k:v[j] for k,v in a['component_totals_raw'].items()},
                    'full_return':a['full_episode_return'][j],
                    'housing_penalty_saturated_fraction_9to13':float((c[89:130]<=0).mean()),
                    'housing_min_m':float(c[89:130].min())})
    out['housing_signal'][str(seed)]=records
print(json.dumps(out,allow_nan=False))
