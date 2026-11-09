"""Independent saved-row and sparse-chain audit. No FK/root/AD/solver rerun.
Original446 and its five-row-only correction are both retained. This audit
checks the corrected dense mesh stencils, original constraint replay and
implicit tangent algebra; it does not admit the infeasible440 input.
"""
from pathlib import Path
from fractions import Fraction
from collections import defaultdict
import json,hashlib,datetime,bisect
import numpy as np
from scipy import sparse
from scipy.spatial import ConvexHull
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');I=D/'clock_row_preparation_446';V=D/'clock_row_preparation_v2_446';O=D/'clock_row_audit_447';EPS=2.**-23
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert read(D/'clock_row_preparation_exit_446.json')['actual_exit']==0 and read(D/'clock_row_finalize_exit_v2_446.json')['actual_exit']==0
 for folder in [I,V]:
  for p,h in read(folder/'SHA256.json').items():assert sha(folder/p)==h,p
 for p,h in read(V/'source_evidence.json').items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);a=dict(np.load(V/'row_arrays.npz'));old=dict(np.load(I/'row_arrays.npz'));M=sparse.load_npz(V/'all_rows_matrix.npz');J=sparse.load_npz(V/'global_jacobian.npz');origM=sparse.load_npz(I/'all_rows_matrix.npz');origJ=sparse.load_npz(I/'global_jacobian.npz');rowmap=np.array(read(V/'original_row_map.json'))
 assert (M-origM[rowmap]).nnz==0 and (J-origJ[rowmap]).nnz==0 and np.array_equal(rowmap[:29280],np.arange(29280))
 for k in ['constant','scale','keep','soft','groups','rows']:assert np.array_equal(a[k],old[k][rowmap]),k
 assert origM.shape[0]-M.shape[0]==5
 g=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));pr=dict(np.load(D/'clock_transcription_preparation_444/primal_replay.npz'));s=dict(np.load(D/'contact_lift_440/states.npz'));ini=dict(np.load(D/'lattice_path_430/initial.npz'));rr=read(D/'contact_lift_440/records.json');times=[Fraction(int(x),int(y)) for x,y in read(D/'clock_transcription_preparation_444/exact_times.json')];ls=g['label_to_state'];canonical=g['canonical'];K=len(canonical);F=np.c_[pr['X'][canonical],pr['Y'][canonical]]
 values=M@F.ravel()+a['constant'];assert np.array_equal(values,a['rows'])
 errors={'original_primal_rows':float(abs(a['old_raw']-s['node_raw']).max()),'prior_primal_rows':float(abs(a['old_prior']-s['node_prior']).max())}
 keep=np.load(D/'lattice_path_430/final.npz')['keep'].astype(bool);buf=np.r_[values[:27708][keep]/a['scale'][:27708][keep],values[27708:29280]]
 errors['old_buffered_rows']=float(abs(buf-s['node_buffered']).max());assert max(errors.values())<2e-10
 assert np.array_equal(keep,a['keep'][:27708]) and np.all(a['keep'][27708:])
 # Independently enumerate EVERY newly added row. This compares all sparse
 # coefficients, constants, numerical scales and hard/soft labels, not only
 # a minimum or a random selection of rows.
 pos=29280;maxcoef=0.;maxconstant=0.;maxscale=0.
 def check(group,terms,constant,scale,soft=False):
  nonlocal pos,maxcoef,maxconstant,maxscale
  assert a['groups'][pos]==group and bool(a['soft'][pos])==soft,(pos,group,a['groups'][pos])
  expected=defaultdict(float)
  for block,feature,value in terms:expected[int(block)*75+feature]+=value
  expected={k:v for k,v in expected.items() if v!=0};q=M.getrow(pos);actual={int(k):float(v) for k,v in zip(q.indices,q.data) if v!=0}
  assert set(actual)==set(expected),(pos,group,actual,expected)
  e=max([abs(actual[k]-expected[k]) for k in expected]+[0.]);maxcoef=max(maxcoef,e);maxconstant=max(maxconstant,abs(a['constant'][pos]-constant));maxscale=max(maxscale,abs(a['scale'][pos]-scale));pos+=1
 lo=ini['command_lo'];hi=ini['command_hi'];rates=ini['rates'];ulp=ini['ulp'];eff=ini['effort'];dep=[0,1]+list(range(4,16))
 for b,label in enumerate(canonical):
  for j in range(16):
   check('mesh_command_lower',[(b,39+j,1)],-lo[j]-.03*EPS,.03);check('mesh_command_upper',[(b,39+j,-1)],hi[j]-.03*EPS,.03)
   check('mesh_effort_lower',[(b,55+j,1)],eff[j]*(1-EPS),eff[j]);check('mesh_effort_upper',[(b,55+j,-1)],eff[j]*(1-EPS),eff[j])
  if not g['fixed'][b]:
   for j in dep:
    l=-1. if j<2 else ini['lower'][0,j];u=1. if j<2 else ini['upper'][0,j]
    check('mesh_planning_lower',[(b,j,1)],-l-EPS,1.);check('mesh_planning_upper',[(b,j,-1)],u-EPS,1.)
  points=s['points'][label];points=points[:4] if rr[label]['stance']==['left'] else points;points=points[points[:,2]<points[:,2].min()+.003]
  for h in ConvexHull(points[:,:2]).equations:check('mesh_support',[(b,72,-h[0]),(b,73,-h[1])],-h[2]-.005-8*np.finfo(np.float32).eps-.005*EPS-2e-12,.005)
  check('mesh_gap_nonnegative',[(b,71,1)],1e-8-.012*EPS,.012);t=times[label]
  if 0<t<times[1450]:check('mesh_prehover',[(b,71,1)],-.012-.012*EPS,.012)
  if times[1451]<t<=times[1456]:check('mesh_posthover',[(b,71,-1)],.012-.012*EPS,.012)
 rep=g['ordered_representative'];ts=[times[i] for i in rep];states=ls[rep]
 for k in range(len(ts)-1):
  u,v=int(states[k]),int(states[k+1]);dt=float(ts[k+1]-ts[k])
  if u==v:continue
  if ts[k+1]<=times[1456]:
   for j in [2,3]:check('mesh_phase',[(v,j,1),(u,j,-1)],0.,1.)
  for sign,name in [(-1,'mesh_rate_increasing'),(1,'mesh_rate_decreasing')]:
   for j in range(16):check(name,[(v,39+j,sign),(u,39+j,-sign)],dt*(rates[j]*(1-EPS)-4*ulp[j]/.02),dt*rates[j],True)
  if times[1416]<=ts[k] and ts[k+1]<=times[1456]:check('mesh_approach_order',[(u,71,1),(v,71,-1)],0.,.012)
  if times[1451]<=ts[k] and ts[k+1]<=times[1456]:check('mesh_normal_descent',[(v,71,1),(u,71,-1)],.006*dt*(1-EPS),.006*dt)
 for k in range(675):
  for sign,name in [(-1,'clock_target_upper'),(1,'clock_target_lower')]:
   for j in range(16):check(name,[(int(ls[k+1]),39+j,10*sign),(int(ls[k]),39+j,-9*sign)],(hi[j] if sign==-1 else -lo[j])-32*ulp[j]-.03*EPS,.03,True)
 def interp(t):
  if t<=ts[0]:return [(int(states[0]),1.)]
  if t>=ts[-1]:return [(int(states[-1]),1.)]
  k=bisect.bisect_right(ts,t)-1;a=float((t-ts[k])/(ts[k+1]-ts[k]));return [(int(states[k]),1-a),(int(states[k+1]),a)]
 for t in ts+[x+Fraction(1,50) for x in ts]:
  coeff=defaultdict(float)
  for mult,q in [(10,t),(-9,t-Fraction(1,50))]:
   for n,v in interp(q):coeff[n]+=mult*v
  for sign,name in [(-1,'mesh_window_upper'),(1,'mesh_window_lower')]:
   for j in range(16):check(name,[(n,39+j,sign*v) for n,v in coeff.items()],(hi[j] if sign==-1 else -lo[j])-32*ulp[j]-.03*EPS,.03,True)
 assert pos==M.shape[0] and max(maxcoef,maxconstant,maxscale)<2e-12,(pos,M.shape,maxcoef,maxconstant,maxscale)
 c=dict(np.load(I/'coordinate_map.npz'));L=g['canonical_J'];d=c['dependent'];control=c['control'];T=np.zeros_like(c['T']);T[:,control,:]=np.eye(10);T[:,d,:]=-np.linalg.solve(L[:,:14][:,:,d],L[:,:14][:,:,control]);G=np.concatenate([T,np.einsum('noi,nic->noc',L,T)],axis=1)
 errors['tangent_map']=float(abs(T-c['T']).max());errors['full_feature_map']=float(abs(G-c['G']).max());assert errors['full_feature_map']<2e-12
 full=sparse.block_diag(list(G),format='csr')[:,c['free']];delta=M@full-J;errors['global_jacobian']=float(abs(delta.data).max()) if delta.nnz else 0.;assert errors['global_jacobian']<2e-12
 E=sparse.load_npz(I/'equality_matrix.npz');EJ=sparse.load_npz(I/'equality_jacobian.npz');eq=dict(np.load(I/'equality_arrays.npz'));assert np.array_equal(eq['values'],E@F.ravel()+eq['constant']);assert (E@full-EJ).nnz==0;assert eq['optimizer_keep'].sum()==1366
 fixture=dict(np.load(I/'new_root_directional.npz'));errors['new_root_directional_replay']=float(abs(fixture['finite']-fixture['expected']).max());assert errors['new_root_directional_replay']<2e-6
 roots=[json.loads(x) for x in (I/'new_root_fixtures.jsonl').read_text().splitlines()];assert len(roots)==2 and all(x['maximum_equation_residual']<=1e-13 for x in roots)
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,arithmetic_only=True,total_inequality_rows=M.shape[0],new_mesh_rows_independently_rebuilt=M.shape[0]-29280,all_new_coefficient_error=maxcoef,all_new_constant_error=maxconstant,all_new_scale_error=maxscale,errors=errors,original_rows_replayed=27708,original_bounds_replayed=1572,removed_only_five_extra64_to65_gap_edges=True,old_rows_and_buffers_unchanged=True,seed_is_infeasible=True,full_global_solver_not_frozen=True,optimizer_launch_allowed=False,physical_admission=False,FK_calls=0,root_calls=0,AD_calls=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
