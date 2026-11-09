import sys,json
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;sys.path.insert(0,str(P.with_name('box_armchair_lean_20261104_v1')))
from plan import ROOT,LeanPlan
from wrench import WrenchObserver
from humanoid_sitting.shared_frame import BodyModel
r=ROOT/'outputs/box_armchair_lean_20261104_v1/trial_0'
frame=np.load(r/'frame_0.npz');z=np.load(r/'trajectory.npz');data=json.loads((r/'plan.json').read_text())
bank=np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning/bank.npz')
p=LeanPlan(BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json'),frame['q'],frame['gravity'],frame['geometry'],frame['selected_points'],bank['reset_command'][0])
obs=WrenchObserver(p,data['nodes']); nodes=data['nodes'];t=data['times']
torque=z['physical_torque'][:,0][:,p.ids].sum(1)
times=(1+np.arange(1000))*.02
ix=np.abs(times[:,None]-t).argmin(1)
res=np.array([obs.tables[k][:,0]-v for k,v in zip(ix,torque)])
obs.bias=np.median(res[(times>8.5)&(times<9.5)],axis=0)
est=np.array([obs.estimate(k,v) for k,v in zip(ix,torque)])
F=z['force'][:,0,3:5].sum(1);n=np.array([.982186,0,-.187911]);up=np.array([-n[2],0,n[0]])
for l,h in [(0,.5),(6,8),(8.5,9.5),(10.5,11),(12,13),(15,20)]:
 mask=(times>=l)&(times<h);print(l,h,'estimated mean',est[mask].mean(0),'measured',(F[mask]@np.stack([n,up]).T).mean(0))
print('bias',obs.bias,'J',obs.tables[-1])
