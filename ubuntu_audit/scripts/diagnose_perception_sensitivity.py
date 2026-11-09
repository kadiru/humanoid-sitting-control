"""Causal offline feature substitutions in the frozen recurrent trajectory head.

No simulation success claim: this diagnoses changes in generated body targets.
"""
import argparse,json
from pathlib import Path
import numpy as np,torch
from humanoid_sitting.deployment import DeploymentStudent,seat_vector
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--clean',required=True);p.add_argument('--noisy',required=True);p.add_argument('--output',required=True);a=p.parse_args();torch.set_num_threads(8);m,_=DeploymentStudent.load(a.checkpoint,'cuda');c=json.loads((Path(a.clean)/'estimates.json').read_text());n=json.loads((Path(a.noisy)/'estimates.json').read_text());ids=[0,1,2,9,10,11,18,19,20,27,28,29];clean=torch.tensor([seat_vector(c[i]) for i in ids],device='cuda');noisy=torch.tensor([seat_vector(n[i]) for i in ids],device='cuda');rows=[]
with torch.inference_mode():
 x=m.mean[None,None,:].expand(12,1000,-1).clone();x[:,:,98:108]=clean[:,None,:];x[:,:,108]=1;reference,_=m(x)
 for name,columns in [('all_geometry',list(range(10))),('confidence',[8]),('position_and_height',[0,1,2,3]),('visible_dimensions',[4,5]),('heading',[6,7])]:
  inp=x.clone()
  for k in columns:inp[:,:,98+k]=noisy[:,None,k]
  targets,_=m(inp);body=[i for i in range(23) if i not in (15,16)];error=(targets-reference)[:,:,body].cpu().numpy();rows.append(dict(substitution=name,body_target_rmse_rad=float(np.sqrt((error**2).mean())),maximum_body_target_difference_rad=float(abs(error).max()),first2s_maximum_hip_target_difference_rad=float((targets-reference)[:,:100,:2].abs().max())))
result=dict(scope='Offline body-head sensitivity only; no dynamics, safety or hardware execution claim',checkpoint=a.checkpoint,measured_clean_and_noisy_geometry=True,rows=rows,training_confidence_normalizer_std=float(m.std[106]),clean_confidence=[c[i]['confidence'] for i in ids],noisy_confidence=[n[i]['confidence'] for i in ids]);Path(a.output).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
