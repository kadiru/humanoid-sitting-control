"""Independent COM/force/support and analytical kinematic audit of new371 preparation.
No construction/projection/optimization rerun; no runtime or physical admission.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation
from humanoid_sitting.shared_frame import BodyModel,apply,transform
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'support_preparation_audit_372'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cross(a,b):return np.array([a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]])
def skew(a):return np.array([[0,-a[2],a[1]],[a[2],0,-a[0]],[-a[1],a[0],0]])
def main():
 O.mkdir(exist_ok=False);r=read(D/'support_preparation_371/preparation.json');z=np.load(D/'support_preparation_371/fixture.npz');a={k:z[k] for k in z.files}
 for f,h in r['source_input_sha256'].items():assert sha(f)==h,f
 m=BodyModel(D/'cooked_model_50/model.json');ine=read(D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json');base=read(D/'clearance_lift_path_330/results.json');old=base['dense']['path'];seed=read(D/'multiknot_path_344/results.json');nodes=seed['dense']['command_nodes'];mass=np.array(ine['mass']);W=mass.sum()*9.81;sides=['left','right'];legs=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['hip_pitch','hip_roll','hip_yaw','knee','ankle_pitch','ankle_roll']];arms=[m.names.index(s+'_'+j+'_joint') for s in sides for j in ['shoulder_pitch','elbow_pitch']];moving=legs+arms;kp=np.array([200,150,150,200,160,160]*2+[40]*4);joints={j['name']:j for j in m.joints};parent={j['body1'][0]:j['body0'][0] for j in m.joints if j['body0']};dep=np.array([0,1]+list(range(4,16)))
 def below(body,ancestor):
  while True:
   if body==ancestor:return True
   if body not in parent:return False
   body=parent[body]
 R0=Rotation.from_quat(old[0]['root'][3:]).as_matrix();axis=Rotation.from_matrix(R0.T@Rotation.from_quat(base['endpoint']['root'][3:]).as_matrix()).as_rotvec();dz=base['endpoint']['root'][2]-old[0]['root'][2]
 mx=dict(force_particular=0.,equilibrium=0.,com_target=0.,full_fk=0.,virtual_work_Nm=0.,dependent_direction=0.,raw_direction=0.,support=0.);support=[];sk=[];count=0
 for i in range(114):
  k=i+1;cnt=int(a['counts'][i]);P=a['points'][i,:cnt];ss=nodes[k]['corner_sides'];active=[j for j,s in enumerate(ss) if not 16<=k<=64 or s=='left'];A=np.vstack([np.ones(len(active)),P[active,:2].T]);wanted=A.T@np.linalg.solve(A@A.T,np.array([[0,0],[.1,0],[0,.1]]));mx['force_particular']=max(mx['force_particular'],float(abs(wanted-a['wcom'][i,active]).max()));assert max(abs(A@a['null'][i,active]).ravel())<1e-12;assert abs(A@a['wcom'][i,active]-np.array([[0,0],[.1,0],[0,.1]])).max()<1e-12
  assert np.array_equal(a['base_weights'][i,:cnt],np.array(nodes[k]['corner_fractions']))
  if 16<=k<=64:
   inactive=[j for j,s in enumerate(ss) if s=='right'];assert np.count_nonzero(a['wcom'][i,inactive])==0 and np.count_nonzero(a['null'][i,inactive])==0 and np.count_nonzero(a['base_weights'][i,inactive])==0
  ot=transform(Rotation.from_quat(old[k]['root'][3:]).as_matrix(),np.array(old[k]['root'][:3]));of=m.fk(np.array(old[k]['q']));verts=np.concatenate([apply(ot@of[s+'_ankle_roll_link'],m.vertices[s+'_ankle_roll_link']) for s in old[k]['stance']]);sole=verts[verts[:,2]<verts[:,2].min()+.003]
  if k<104:support.extend(ConvexHull(sole[:,:2]).equations);sk.extend([i]*len(ConvexHull(sole[:,:2]).equations))
  for prefix in ['', 'test_']:
   q=a[prefix+'Q'][i];T=a[prefix+'T'][i];X=a[prefix+'X'][i];Y=a[prefix+'Y'][i];fk=m.fk(q);points={b:apply(T@fk[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])};C=sum(mm*points[b] for mm,b in zip(mass,ine['body_names']))/mass.sum();mx['full_fk']=max(mx['full_fk'],float(abs(C-a[prefix+'C'][i]).max()));mx['com_target']=max(mx['com_target'],float(abs(C[:2]-np.array(old[k]['com'][:2])-.1*X[21:23]).max()));w=a['base_weights'][i]+a['null'][i]@X[16:21]+a['wcom'][i]@X[21:23];assert abs(w-Y[46:54]).max()<1e-13
   if k<=103:mx['equilibrium']=max(mx['equilibrium'],float(max(abs(w.sum()-1),abs(w[:cnt]@P[:,:2]-C[:2]).max())))
   local=[np.linalg.inv(T@fk[s+'_ankle_roll_link'])@np.r_[pp,1.] for s,pp in zip(ss,P)]
   for j,torque in zip(moving,Y[30:46]):
    work=[]
    for sign in [-1,1]:
     qq=q.copy();qq[j]+=sign*1e-5;ff=m.fk(qq);V=sum(mm*9.81*apply(T@ff[b],cc)[2] for mm,b,cc in zip(mass,ine['body_names'],ine['local_com']));F=sum(W*ww*(T@ff[s+'_ankle_roll_link']@ll)[2] for ww,s,ll in zip(w,ss,local));work.append(V-F)
    mx['virtual_work_Nm']=max(mx['virtual_work_Nm'],abs((work[1]-work[0])/2e-5-torque))
   count+=1
  if k<=103:
   q=a['Q'][i];T=a['T'][i];fk=m.fk(q);points={b:apply(T@fk[b],cc) for b,cc in zip(ine['body_names'],ine['local_com'])};geo={}
   for j in legs+arms:
    name=m.names[j];jj=joints[name];J=T@fk[jj['body0'][0]]@m.frames[name][0];geo[j]=(J[:3,'XYZ'.index(jj['axis'])],J[:3,3],jj['body1'][0])
   J=np.zeros((14,18))
   for col in range(16):
    rootv=np.eye(3)[:,col] if col<2 else np.array([0,0,dz]) if col==14 else np.zeros(3);rotv=T[:3,:3]@axis if col==15 else np.zeros(3);activej=[(legs[col-2],1.)] if 2<=col<14 else [(j,-.35) for j in arms] if col==14 else []
    def velocity(body,P):
     v=rootv+cross(rotv,P-T[:3,3]);w=rotv.copy()
     for j,weight in activej:
      ax,origin,child=geo[j]
      if below(body,child):v+=weight*cross(ax,P-origin);w+=weight*ax
     return v,w
    for t,s in enumerate(sides):
     body=s+'_ankle_roll_link';F=T@fk[body];v,w=velocity(body,F[:3,3]);J[t*6:t*6+3,col]=v;dR=(ot@of[body])[:3,:3].T@skew(w)@F[:3,:3];J[t*6+3:t*6+6,col]=.1*np.array([dR[2,1]-dR[1,2],dR[0,2]-dR[2,0],dR[1,0]-dR[0,1]])
    J[12:,col]=(sum(mm*velocity(b,points[b])[0] for mm,b in zip(mass,ine['body_names']))/mass.sum())[:2]
   J[12:,16:]=-.1*np.eye(2);tangent=np.linalg.lstsq(J[:,:14],-J[:,14:],rcond=None)[0];d=a['direction'][:1026].reshape(114,9)[i,:4];want=tangent@d;fd=(a['plus_X'][i,dep]-a['minus_X'][i,dep])/(2*float(a['h']));mx['dependent_direction']=max(mx['dependent_direction'],float(abs(want-fd).max()))
 support=np.array(support);assert np.array_equal(np.array(sk),a['support_k']);mx['support']=float(abs(support-a['support_h']).max());assert mx['support']<1e-12
 # Independently reconstruct row scaling/partition without calling371/365 helpers.
 scale=np.r_[np.tile(np.r_[np.full(32,.03),a['effort'],a['effort'],np.ones(16)],114),np.tile(np.r_[[.1,.1],np.ones(12),[.1,.1],np.ones(12)],103),np.ones(208),np.tile(a['rates']*.02,114*2),np.full(3648,.03*.02),np.full(98+5+104+39,.012),np.full(5,.006*.02),13.5,np.full(len(support),.005)];assert np.array_equal(scale,a['scale'])
 hard=np.ones(len(scale),bool)
 for i in range(114):hard[i*80:i*80+32]=False
 hard[12212:19508]=False;hard[19759]=False
 assert np.array_equal(hard,a['hard'])
 keep=np.ones(len(scale),bool)
 for i in range(114):
  for j in range(8):
   if not np.count_nonzero(a['null'][i,j]) and not np.count_nonzero(a['wcom'][i,j]):
    assert 0<=a['base_weights'][i,j]<=1
    keep[i*80+64+j]=False;keep[i*80+72+j]=False
 assert np.array_equal(keep,a['keep']);slack=float(a['slack']);aug=np.array([float(v)/float(s)+(0 if hh else slack) for v,s,hh,kk in zip(a['ineq'],scale,hard,keep) if kk]);assert abs(aug-a['augmented']).max()<1e-12
 for mat,an,fd in [(a['E'],a['eq_analytic'],a['eq_fd']),(a['I'],a['ineq_analytic'],a['ineq_fd'])]:
  val=np.array([sum(float(x)*float(y) for x,y in zip(row,a['direction'][a['free']])) for row in mat]);assert abs(val-an).max()<1e-12;mx['raw_direction']=max(mx['raw_direction'],float(abs(val-fd).max()))
 assert max(mx['force_particular'],mx['equilibrium'],mx['com_target'],mx['full_fk'])<1e-10;assert mx['virtual_work_Nm']<1e-6;assert mx['dependent_direction']<2e-6 and mx['raw_direction']<2e-6
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,records=count,finite_work_columns=count*16,analytic_kinematic_records=103,analytic_dependent_and_independent_columns=103*18,all_rows=len(scale),original_rows=19760,added_support_rows=len(support),maximum_errors=mx,removed_rows=int((~keep).sum()),hard_rows=int((hard&keep).sum()),soft_rows=int((~hard&keep).sum()),hard_hover_equality=True,initial_feasibility_not_claimed=True,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0,source_sha256={str(Path(__file__).resolve()):sha(__file__),str(D/'support_preparation_371/preparation.json'):sha(D/'support_preparation_371/preparation.json'),str(D/'support_preparation_371/fixture.npz'):sha(D/'support_preparation_371/fixture.npz')})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({f.name:sha(f) for f in O.iterdir() if f.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
