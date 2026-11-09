"""Independent all-knot geometry and virtual-work replay for lateral-demand225 (all coarse and dense records), retaining failed gates."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull,QhullError
from scipy.spatial.transform import Rotation,Slerp
from humanoid_sitting.shared_frame import BodyModel,apply,transform,level_rotation
from plan_seat_load_targets import clipped_vertices
from low_seat_terminal_contract_42 import terminal_posture_check
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'lateral_demand_audit_226'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);src=D/'lateral_demand_refinement_225';r=read(src/'results.json')
 for p,h in r['source_sha256'].items():assert sha(p)==h,p
 m=BodyModel(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');mass=np.array(ine['mass']);W=mass.sum()*9.81;limits={j['name']:np.deg2rad(j['limits']) for j in m.joints if j['type']=='PhysicsRevoluteJoint'};kp=np.array([200,150,150,200,160,160]*2+[40]*4);eff=np.array([300,300,300,300,20,20]*2+[300]*4);checked=[];maxgeom=maxwork=maxcmd=maxeq=0.
 files=sorted(src.glob('coarse_*.json'))+sorted(src.glob('case_*.json'))
 for f in files:
  row=read(f);c=row['case'];p=row['plan'];x=row['inputs'];ids=p['moving_ids'];fixed=[j for j in range(37) if j not in ids];q0=np.array(x['q']);cmd0=np.array(x['command']);tau0=np.array(p['initial_static_torque_Nm']);n=np.array(p['normal']);d=p['offset'];origin=np.array(p['origin']);R0=level_rotation(np.array(x['gravity']));fk0=m.fk(q0);feet0={s:(R0@fk0[s+'_ankle_roll_link'][:3,3]+origin,R0@fk0[s+'_ankle_roll_link'][:3,:3]) for s in ['left','right']};SR=Rotation.from_euler('y',p['slope_deg'],degrees=True).as_matrix();center=np.array([-.4,0,p['height']]);gates=[]
  for k,e in enumerate(p['path']):
   q=np.array(e['q']);pose=np.array(e['root']);T=transform(Rotation.from_quat(pose[3:]).as_matrix(),pose[:3]);fk=m.fk(q);assert np.array_equal(q[fixed],q0[fixed]);foot={s:apply(T@fk[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in feet0};C=np.average([apply(T@fk[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])],axis=0,weights=mass)
   assert np.max(abs(C-e['com']))<1e-12
   for s in feet0:assert np.array_equal(e['foot_targets'][s]['position'],feet0[s][0]) and np.array_equal(e['foot_targets'][s]['rotation'],feet0[s][1])
   st=np.concatenate(list(foot.values()));sole=st[st[:,2]<st[:,2].min()+.003]
   try:H=ConvexHull(sole[:,:2]).equations;cop=float(min(-(H[:,:2]@C[:2]+H[:,2])))
   except QhullError:cop=-1.
   v=apply(T,m.vertices['pelvis']);dep=v@n+d;patch=dep<dep.min()+.002;loc=(v-center)@SR;pm=float(min(.25-abs(loc[patch,0]).max(),.45-abs(loc[patch,1]).max()));clear=[];floor=[]
   for b,verts in m.vertices.items():
    w=apply(T@fk[b],verts)
    if b not in ['left_ankle_roll_link','right_ankle_roll_link']:floor.append(float(w[:,2].min()))
    if b in ['pelvis','left_hip_yaw_link','right_hip_yaw_link','torso_link','logo_link','left_ankle_roll_link','right_ankle_roll_link']:continue
    v=(w-center)@SR;v[:,0]-=.4;v=clipped_vertices(v)
    if len(v):clear.append((float(v[:,2].min()),b))
   mi=min(clear) if clear else (1.,'none_over_finite_seat');pe=max(np.linalg.norm((T@fk[s+'_ankle_roll_link'])[:3,3]-feet0[s][0]) for s in feet0);re=max(np.linalg.norm(Rotation.from_matrix(feet0[s][1].T@(T@fk[s+'_ankle_roll_link'])[:3,:3]).as_rotvec()) for s in feet0);term=terminal_posture_check(pose)
   val=dict(clearance_m=mi[0],pelvis_gap_m=float(dep.min()),patch_margin_m=pm,foot_cop_margin_m=cop,floor_clearance_m=min(floor),foot_position_error_m=pe,foot_orientation_error_rad=re);ge=max(abs(e[z]-v) for z,v in val.items());assert ge<1e-9;maxgeom=max(maxgeom,ge)
   other=bool(pe<=.003 and re<=np.deg2rad(.5) and cop>=.005 and min(floor)>=.005 and term['passed'] and (dep.min()>=.02 or pm>=.015) and dep.min()>=-1e-8);assert other==e['other_model_passed'];assert bool(other and mi[0]>=.003)==e['model_pose_passed']
   load=e['load_checks'][0];P=np.array(load['corner_world_points']);ww=np.array(load['corner_fractions']);sides=np.array(load['corner_sides']);sf=np.array(load['seat_force_normalized']);contact=np.array(load['seat_contact_point']);eq=max(abs(ww.sum()+sf[2]-1),max(abs((ww[:,None]*P[:,:2]).sum(0)+sf[2]*contact[:2]-C[:2])));maxeq=max(maxeq,float(eq));assert eq<1e-8 and min(ww)>=-1e-10
   if dep.min()>1e-8:assert np.array_equal(sf,np.zeros(3))
   assert abs(sf@n-e['assumed_seat_normal_fraction'])<1e-12
   local=[np.linalg.inv(T@fk[s+'_ankle_roll_link'])@np.r_[a,1.] for a,s in zip(P,sides)];tau=[]
   for j in ids:
    v=[]
    for sign in [-1,1]:
     qq=q.copy();qq[j]+=sign*1e-5;ff=m.fk(qq);v.append(sum(mm*9.81*apply(T@ff[b],cc)[2] for b,mm,cc in zip(ine['body_names'],mass,ine['local_com']))-sum(W*w*(T@ff[s+'_ankle_roll_link']@ll)[2] for w,s,ll in zip(ww,sides,local)))
    tau.append((v[1]-v[0])/2e-5)
   tau=np.array(tau);we=float(max(abs(tau-load['static_torque_Nm'])));assert we<1e-6;maxwork=max(maxwork,we);cmd=np.array(load['command']);expected=cmd0.copy();expected[ids]+=q[ids]-q0[ids]+(tau-tau0)/kp;ce=float(max(abs(cmd-expected)));assert ce<2e-8;maxcmd=max(maxcmd,ce);assert np.array_equal(cmd[fixed],cmd0[fixed])
   cm=min(min(cmd[j]-limits[m.names[j]][0],limits[m.names[j]][1]-cmd[j]) for j in ids);eu=max(abs(np.array(load['static_torque_Nm']))/eff);assert abs(cm-load['min_command_margin_rad'])<1e-12 and abs(eu-load['maximum_effort_utilization'])<1e-12
   cp=[]
   for side in ['left','right']:
    mask=sides==side;pos=(P[mask,:2]*ww[mask,None]).sum(0)/ww[mask].sum();H=ConvexHull(P[mask,:2]).equations;margin=float(min(-(H[:,:2]@pos+H[:,2])));assert abs(margin-load['individual_foot_cop'][side]['margin_m'])<1e-12;cp.append(margin)
   arithmetic=bool(eq<1e-8 and min(ww)>=-1e-10 and cm>=.03 and eu<=1 and min(cp)>=.005-1e-10);assert load['passed']==bool(load['solver_success'] and arithmetic)
   gates.append(dict(knot=k,geometry=e['model_pose_passed'],commands=load['passed'],clearance_mm=mi[0]*1000,individual_cop_mm=[v*1000 for v in cp]))
  cmd=np.array([e['load_checks'][0]['command'] for e in p['path']]);time=float(np.max(abs(np.diff(cmd,axis=0)),axis=1).sum()/.15);assert abs(time-p['summary']['command_polyline_time_s'])<1e-12;assert p['model_passed']==all(z['geometry'] and z['commands'] for z in gates)
  checked.append(dict(file=f.name,case=c,knots=len(p['path']),model_passed=p['model_passed'],failed_geometry=[z['knot'] for z in gates if not z['geometry']],failed_commands=[z['knot'] for z in gates if not z['commands']],minimum_clearance_mm=min(z['clearance_mm'] for z in gates),records=gates));print(c,p['model_passed'],flush=True)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,records=sum(z['knots'] for z in checked),checks=checked,max_geometry_error=maxgeom,max_virtual_work_error_Nm=maxwork,max_command_error_rad=maxcmd,max_equilibrium_residual=maxeq,physics_ticks=0,optimizer_updates=0,physical_admission=False,failed_model_gates_retained=True,source_sha256={str(src/'results.json'):sha(src/'results.json'),str(Path(__file__).resolve()):sha(__file__)})
 revision=read(src/'revision.json')
 for c in [0,6]:
  hist=read(src/f'case_{c}.json')['history'];assert len(hist)<=40 and len(hist)==len(list(src.glob(f'coarse_{c}_*.json')))
  for a in hist:
   xx,yy,rr,pp,zz=a['parameters'];assert -.28<=xx<=-.1 and np.deg2rad(5)<=pp<=np.deg2rad(38)
   old=read(src/f'case_{c}.json')['summary']['original_endpoint_parameters'];assert min(0,old[1])<=yy<=max(0,old[1]) and rr==old[2] and zz==old[4]
 assert len(list(src.glob('case_*.json')))==2
 import torch
 filters=[]
 for c in [0,6]:
  row=read(src/f'case_{c}.json');p=row['plan'];schedule=row['schedule'];assert schedule['checked']
  z=np.load(src/f'schedule_{c}.npz');ids=p['moving_ids'];fixed=[j for j in range(37) if j not in ids];q=torch.tensor(z['initial'],dtype=torch.float32);initial=q.clone();lo=torch.tensor([limits[m.names[j]][0]+.03 for j in ids],dtype=torch.float32);hi=torch.tensor([limits[m.names[j]][1]-.03 for j in ids],dtype=torch.float32);cap=torch.full((len(ids),),.003,dtype=torch.float32)
  for j,i in enumerate(ids):
   if m.names[i] in ['left_ankle_pitch_joint','right_ankle_pitch_joint']:cap[j]=.00225
  maximum=0.
  for target,nom in zip(z['targets'],z['nominal']):
   t=torch.tensor(target,dtype=torch.float32);assert bool(torch.all(t[ids]>=lo)) and bool(torch.all(t[ids]<=hi));before=q.clone();q[ids]=q[ids]+torch.clamp(torch.tensor(.1,dtype=torch.float32)*(torch.clamp(t[ids],lo,hi)-q[ids]),-cap,cap);assert np.array_equal(q.numpy(),nom);assert torch.equal(q[fixed],initial[fixed]);maximum=max(maximum,float(torch.max(torch.abs(q-before))))
  assert np.array_equal(q.numpy(),np.array(p['path'][-1]['load_checks'][0]['command'],np.float32))
  assert len(z['targets'])*.02==schedule['duration_s'] and schedule['duration_s']<=13.5
  filters.append(dict(case=c,ticks=len(z['targets']),exact_torch_cpu_replay=True,fixed_commands_exact=True,endpoint_exact=True,clipped_targets=0,max_applied_step_rad=maximum,duration_s=schedule['duration_s'],synthetic_quiet_load=True,physical_admission=False))
 out['filter_replay']=filters
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
if __name__=='__main__':main()
