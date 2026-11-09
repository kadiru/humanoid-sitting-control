"""UNEXECUTED complete-clock row assembly for the new444 formulation.

All rows are affine in local mechanics outputs and direct coordinates. The
nonlinearity is entirely in the audited local map; root retraction/solver are
separate and not implemented here. Original115-node rows and coordinate bounds
are retained explicitly, with their original scales and all inward reserves.
Additional mesh constraints do not turn finite sampling into a continuous
geometry/implicit-command certificate. No tolerance or physical criterion change.
"""
from pathlib import Path
from fractions import Fraction
import json
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
BUFFER=2.**-23

class Rows:
 def __init__(self,K):self.K=K;self.ri=[];self.ci=[];self.v=[];self.b=[];self.scale=[];self.group=[];self.soft=[]
 def add(self,terms,constant,scale=1.,group='',soft=False):
  i=len(self.b)
  for state,feature,value in terms:
   if value!=0:self.ri.append(i);self.ci.append(int(state)*75+int(feature));self.v.append(float(value))
  self.b.append(float(constant));self.scale.append(float(scale));self.group.append(group);self.soft.append(bool(soft))
 def matrix(self):return sparse.coo_matrix((self.v,(self.ri,self.ci)),shape=(len(self.b),self.K*75)).tocsr()

def make_rows():
 P=D/'clock_transcription_preparation_444';g=dict(np.load(P/'mesh.npz'));pr=dict(np.load(P/'primal_replay.npz'));z=dict(np.load(D/'contact_lift_440/states.npz'));ini=dict(np.load(D/'lattice_path_430/initial.npz'));final=dict(np.load(D/'lattice_path_430/final.npz'));prep=dict(np.load(D/'contact_lift_preparation_438/fixture.npz'));rec=read(D/'contact_lift_440/records.json')
 times=[Fraction(int(a),int(b)) for a,b in read(P/'exact_times.json')];ls=g['label_to_state'];canonical=g['canonical'];K=len(canonical);ids=np.r_[np.arange(100),np.arange(101,116)]+1351;nodes=ls[ids];node_times=[times[i] for i in ids]
 dt=np.array([float(node_times[k+1]-node_times[k]-(Fraction(1,2) if k==99 else 0)) for k in range(114)])
 # Node99->100 includes the nominal hold in wall-clock time; old movement dt
 # excludes that hold. Clock/window rows separately retain it exactly.
 assert np.max(abs(dt-z['node_dt']))<1e-15
 C=39;TAU=55;GAP=71;COM=72;W=16
 lo=ini['command_lo'];hi=ini['command_hi'];rates=ini['rates'];ulp=ini['ulp'];eff=ini['effort'];R=Rows(K)
 # Original20284 constraints, in EXACT historical ordering and physical units.
 for k in range(1,115):
  n=nodes[k]
  for j in range(16):R.add([(n,C+j,1)],-lo[j],group='old_command_lower')
  for j in range(16):R.add([(n,C+j,-1)],hi[j],group='old_command_upper')
  for j in range(16):R.add([(n,TAU+j,-1)],eff[j],group='old_effort_upper')
  for j in range(16):R.add([(n,TAU+j,1)],eff[j],group='old_effort_lower')
  mapping=prep['mapping'][k]
  for j in range(8):R.add([] if mapping[j]<0 else [(n,W+mapping[j],1)],0,group='old_force_lower')
  for j in range(8):R.add([] if mapping[j]<0 else [(n,W+mapping[j],-1)],1,group='old_force_upper')
 dep=np.array([0,1]+list(range(4,16)))
 for k in range(1,104):
  for j in dep:R.add([(nodes[k],j,1)],-ini['lower'][k-1,j],group='old_planning_lower')
  for j in dep:R.add([(nodes[k],j,-1)],ini['upper'][k-1,j],group='old_planning_upper')
 for k in range(104):
  for j in [2,3]:R.add([(nodes[k+1],j,1),(nodes[k],j,-1)],0,group='old_phase')
 for sign,name in [(-1,'old_rate_increasing'),(1,'old_rate_decreasing')]:
  for k in range(114):
   for j in range(16):R.add([(nodes[k+1],C+j,sign),(nodes[k],C+j,-sign)],rates[j]*dt[k],group=name,soft=True)
 for sign,name in [(-1,'old_ramp_upper'),(1,'old_ramp_lower')]:
  for k in range(114):
   for j in range(16):R.add([(nodes[k+1],C+j,sign*(dt[k]+.18)),(nodes[k],C+j,-sign*.18)],(hi[j] if sign==-1 else -lo[j])*dt[k],group=name,soft=True)
 for k in range(1,99):R.add([(nodes[k],GAP,1)],-.012,group='old_prehover_gap')
 for k in range(100,105):R.add([(nodes[k],GAP,-1)],.012,group='old_posthover_gap')
 for k in range(1,105):R.add([(nodes[k],GAP,1)],1e-8,group='old_gap_nonnegative')
 for k in range(65,104):R.add([(nodes[k],GAP,1),(nodes[k+1],GAP,-1)],0,group='old_approach_order')
 for k in range(99,104):R.add([(nodes[k],GAP,-1),(nodes[k+1],GAP,1)],.006*dt[k],group='old_normal_descent')
 R.add([],0,group='old_time_identity')
 for H,k in zip(ini['support_h'],ini['support_k']):R.add([(nodes[1+k],COM,-H[0]),(nodes[1+k],COM+1,-H[1])],-H[2]-.005,group='old_support')
 assert len(R.b)==20284
 # Any linear command-polyline target at arbitrary exact rational query time
 # has a sparse pair of coefficients. No event rounding or sample snapping.
 vertex_times=[times[i] for i in range(1351,1467)];vertex_states=ls[1351:]
 def coeff(t,ts,states):
  import bisect
  if t<=ts[0]:return {int(states[0]):1.}
  if t>=ts[-1]:return {int(states[-1]):1.}
  k=bisect.bisect_right(ts,t)-1;a=float((t-ts[k])/(ts[k+1]-ts[k]));out={}
  for s,v in [(int(states[k]),1-a),(int(states[k+1]),a)]:out[s]=out.get(s,0)+v
  return out
 def target_coeff(t,ts,states):
  out={}
  for mult,qq in [(10.,t),(-9.,t-Fraction(1,50))]:
   for s,v in coeff(qq,ts,states).items():out[s]=out.get(s,0)+mult*v
  return out
 #398 interleaves each vertex and its shifted query, clamped at the last
 #vertex. Preserve that older ordering; the NEW mesh envelope below also
 #checks the post-domain breakpoints of its constant extension explicitly.
 for t in [min(t+shift,vertex_times[-1]) for t in vertex_times for shift in [Fraction(0),Fraction(1,50)]]:
  cc=target_coeff(t,vertex_times,vertex_states)
  for sign,name in [(-1,'old_window_upper'),(1,'old_window_lower')]:
   for j in range(16):R.add([(n,C+j,sign*v) for n,v in cc.items()],hi[j] if sign==-1 else -lo[j],group=name,soft=True)
 assert len(R.b)==27708
 oldM=R.matrix();oldb=np.array(R.b);oldraw=oldM@np.c_[pr['X'][canonical],pr['Y'][canonical]].ravel()+oldb
 # Exactly the old unit scales/tightening, not new per-row physical criteria.
 tighten=ini['tighten'][:20284].copy()
 for off in [12212,14036]:tighten[off:off+1824]=(dt[:,None]/.02*4*ulp).ravel()
 for off in [15860,17684]:tighten[off:off+1824]=(dt[:,None]*32*ulp).ravel()
 tighten=np.r_[tighten,np.tile(np.r_[32*ulp,32*ulp],232)]
 oldscale=final['scale'];oldkeep=final['keep'].astype(bool)
 # Store every old row, even structural constants. Only the audited constant
 # contact rows and time identity are excluded from the future optimizer.
 old_prior=oldraw-tighten
 R.b=(oldb-tighten-BUFFER*oldscale).tolist();R.scale=oldscale.tolist()
 # Old786-coordinate bounds, reconstructed affinely from persistent forces
 # and actual model COM. No null-coordinate fit or force modification.
 frozen=read(D/'clearance_lift_path_330/results.json')['dense']['path'];vterms=[];vconst=[]
 for k in range(1,115):
  n=nodes[k];cs=np.array(frozen[k]['com'][:2]);mapping=prep['mapping'][k]
  vterms.extend([[(n,2,1)],[(n,3,1)],[(n,COM,10)],[(n,COM+1,10)]])
  vconst.extend([0.,0.,-10*cs[0],-10*cs[1]])
  for j in range(5):
   B=ini['null'][k-1,:,j];wc=B@ini['wcom'][k-1];terms=[(n,W+mapping[h],B[h]) for h in range(8) if mapping[h]>=0]
   terms.extend([(n,COM,-10*wc[0]),(n,COM+1,-10*wc[1])]);vterms.append(terms);vconst.append(float(-B@ini['base_weights'][k-1]+10*wc@cs))
 shape=ini['shape_free'];bounds=ini['bounds'][:1026][shape]
 for sign,name in [(1,'old_coordinate_lower'),(-1,'old_coordinate_upper')]:
  for j,k in enumerate(shape):R.add([(n,f,sign*v) for n,f,v in vterms[k]],sign*vconst[k]+(-bounds[j,0] if sign==1 else bounds[j,1])-BUFFER,group=name)
 assert len(R.b)==29280
 # New dense local constraints. Immutable poses do not acquire a new planning
 # margin; original104..114 bounds were fixed, and are still checked exactly.
 for b,label in enumerate(canonical):
  for j in range(16):
   R.add([(b,C+j,1)],-lo[j]-.03*BUFFER,.03,'mesh_command_lower')
   R.add([(b,C+j,-1)],hi[j]-.03*BUFFER,.03,'mesh_command_upper')
   R.add([(b,TAU+j,1)],eff[j]*(1-BUFFER),eff[j],'mesh_effort_lower')
   R.add([(b,TAU+j,-1)],eff[j]*(1-BUFFER),eff[j],'mesh_effort_upper')
  if not g['fixed'][b]:
   for j in dep:
    low=-1 if j<2 else ini['lower'][0,j];high=1 if j<2 else ini['upper'][0,j]
    R.add([(b,j,1)],-low-BUFFER,1,'mesh_planning_lower');R.add([(b,j,-1)],high-BUFFER,1,'mesh_planning_upper')
  pts=z['points'][label];stance=rec[label]['stance'];pp=pts[:4] if stance==['left'] else pts;pp=pp[pp[:,2]<pp[:,2].min()+.003];H=ConvexHull(pp[:,:2]).equations
  for h in H:R.add([(b,COM,-h[0]),(b,COM+1,-h[1])],-h[2]-.005-8*np.finfo(np.float32).eps-.005*BUFFER-2e-12,.005,'mesh_support')
  R.add([(b,GAP,1)],1e-8-.012*BUFFER,.012,'mesh_gap_nonnegative')
  t=times[label]
  if t>0 and t<times[1450]:R.add([(b,GAP,1)],-.012-.012*BUFFER,.012,'mesh_prehover')
  if times[1451]<t<=times[1456]:R.add([(b,GAP,-1)],.012-.012*BUFFER,.012,'mesh_posthover')
 # All exact ordered mesh edges, not only the original115 knots. New phase
 # monotonicity has its physical zero bound; old node8eps+BUFFER is unchanged.
 rep=g['ordered_representative'];ts=[times[i] for i in rep];os=ls[rep]
 for i in range(len(ts)-1):
  a,b=int(os[i]),int(os[i+1]);delta=float(ts[i+1]-ts[i])
  if a==b:continue
  if ts[i+1]<=times[1456]:
   for j in [2,3]:R.add([(b,j,1),(a,j,-1)],0,1,'mesh_phase')
  for sign,name in [(-1,'mesh_rate_increasing'),(1,'mesh_rate_decreasing')]:
   for j in range(16):R.add([(b,C+j,sign),(a,C+j,-sign)],delta*(rates[j]*(1-BUFFER)-4*ulp[j]/.02),delta*rates[j],name,True)
  if ts[i]>=times[1415] and ts[i+1]<=times[1456]:R.add([(a,GAP,1),(b,GAP,-1)],0,.012,'mesh_approach_order')
  if ts[i]>=times[1451] and ts[i+1]<=times[1456]:R.add([(b,GAP,1),(a,GAP,-1)],.006*delta*(1-BUFFER),.006*delta,'mesh_normal_descent')
 # Full675 real inverse-filter targets, plus ALL extrema of the new command
 # mesh's linear interpolation. Actual float32 and implicit lift remain later.
 for i in range(675):
  for sign,name in [(-1,'clock_target_upper'),(1,'clock_target_lower')]:
   for j in range(16):R.add([(int(ls[i+1]),C+j,sign*10),(int(ls[i]),C+j,-sign*9)],(hi[j] if sign==-1 else -lo[j])-32*ulp[j]-.03*BUFFER,.03,name,True)
 for t in ts+[t+Fraction(1,50) for t in ts]:
  cc=target_coeff(t,ts,os)
  for sign,name in [(-1,'mesh_window_upper'),(1,'mesh_window_lower')]:
   for j in range(16):R.add([(n,C+j,sign*v) for n,v in cc.items()],(hi[j] if sign==-1 else -lo[j])-32*ulp[j]-.03*BUFFER,.03,name,True)
 keep=np.r_[oldkeep,np.ones(len(R.b)-27708,dtype=bool)]
 return dict(rows=R,old_matrix=oldM,old_constant=oldb,old_raw=oldraw,old_prior=old_prior,old_scale=oldscale,old_tightening=tighten,keep=keep,dt=dt,graph=g,primal=pr,features=np.c_[pr['X'][canonical],pr['Y'][canonical]],node_states=nodes,canonical=canonical)
