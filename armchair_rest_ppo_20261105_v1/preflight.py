"""CPU-only contracts, preserved-demo provenance and fixed center plan checks."""
from pathlib import Path
import sys
import json,time
import numpy as np
import torch
from common import ROOT,read,write,sha
from plan import LeanPlan
from humanoid_sitting.shared_frame import BodyModel
from policy import command
HERE=Path(__file__).resolve().parent;OUT=ROOT/'outputs/armchair_rest_ppo_20261105_v1';OUT.mkdir(exist_ok=True)
inventory={}
for name,count in [('sitting',64),('leaning',16)]:
    folder=ROOT/'outputs/full_body_contact_imitation_v4/dataset'/name
    with np.load(folder/'bank.npz') as z:
        assert int(z['admitted'].sum())==count
        inventory[name]=dict(complete_cases=len(z['admitted']),admitted=count,bank_sha256=sha(folder/'bank.npz'),source=str(folder))
original=read(ROOT/'outputs/motion_prior_ppo_20261029_v1/prepared/preparation.json')
assert original['source'][str(ROOT/'outputs/full_body_contact_imitation_v4/dataset/sitting/bank.npz')]==inventory['sitting']['bank_sha256']
write(OUT/'preserved_demonstrations.json',inventory)
with np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning/bank.npz') as z:prev=z['reset_command'][0]
with np.load(ROOT/'outputs/box_armchair_lean_20261104_v4/trial_0/frame_0.npz') as z:cap=dict(z)
model=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json');plans=read(OUT/'preflight.json')['plans'] if '--pelvis-only' in sys.argv else {}
for profile in (['rear_pelvis'] if '--pelvis-only' in sys.argv else ['rear_back','rear_pelvis']):
    before=time.monotonic()
    try:
        p=LeanPlan(model,cap['q'],cap['gravity'],cap['geometry'],cap['selected_points'],prev,profile=profile)
        nodes,t=p.solve();assert t[-1]<14.4
        plans[profile]=dict(planning_passed=True,duration_s=float(t[-1]),final_q=nodes[-1]['q'],final_height=nodes[-1]['height'],final_pitch=nodes[-1]['pitch'],cpu_seconds=time.monotonic()-before)
    except ValueError as e:plans[profile]=dict(planning_passed=False,error=repr(e),cpu_seconds=time.monotonic()-before)
# Same action envelope and exact conversion at native float32 precision.
pr=torch.tensor(prev[:23])[None];goal=pr+torch.linspace(-.1,.1,23)[None]
desired=pr+(goal-pr).clamp(-.003*.99999,.003*.99999)/.1
meta=read(ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning/metadata.json')
got=command(pr,desired,torch.tensor(meta['lower']),torch.tensor(meta['upper']))
assert ((got-pr).abs()<=.0030002).all()
result=dict(passed=plans['rear_back']['planning_passed'],plans=plans,action_contract_passed=True,original_demonstrations_intact=True,physics_steps=0,PPO_updates=0,scope='Saved center planning only; new starts and physical support remain unvalidated.')
write(OUT/'preflight.json',result);print(json.dumps(result),flush=True)
