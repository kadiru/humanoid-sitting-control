from plan import *
from humanoid_sitting.shared_frame import BodyModel
folder=ROOT/'outputs/furniture_perception_repair_20261029_v1/robot_myd1WSucAz_rear_back'
cap=np.load(folder/'frame_0.npz')
bank=np.load(ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning/bank.npz')
m=BodyModel(ROOT/'runs/g1_kinematics_geometry_04/model.json')
p=LeanPlan(m,cap['q'].astype(float),cap['gravity'].astype(float),cap['geometry'],cap['selected_points'],bank['reset_command'][0])
print('Start',p.geometry(p.start),'band',p.band,'sole',p.foot_bounds,flush=True)
nodes,t=p.solve()
for i in [0,8,16,24,32,42,52,64]: print(i,round(t[i],2),nodes[i],flush=True)
