"""ONE new gap-coordinate and normal-drop preparation; no complete path solve."""
from pathlib import Path
from fractions import Fraction as F
import json,datetime
import numpy as np
from scipy import sparse
from low_seat_clock_transcription_444 import D,sha,load_inputs,Mechanics
from low_seat_gap_coordinates_466 import *
O=D/'gap_coordinate_preparation_466';Q=D/'clock_row_preparation_446';V=D/'clock_row_preparation_v2_446';P=D/'clock_transcription_preparation_444'
def main():
 O.mkdir(exist_ok=False)
 mesh=dict(np.load(P/'mesh.npz'));a=dict(np.load(V/'row_arrays.npz'));cm=dict(np.load(Q/'coordinate_map.npz'));J=mesh['canonical_J'];X=a['features'][:,:24];Y=a['features'][:,24:];K=len(X)
 times=[F(*v) for v in json.loads((P/'exact_times.json').read_text())];ls=mesh['label_to_state'];rep=mesh['ordered_representative'];ordered=[(times[int(i)],int(ls[i])) for i in rep];start=times[1451];end=times[1456]
 normal=[(t,b) for t,b in ordered if start<=t<=end];ns=np.array([b for t,b in normal]);nt=[t for t,b in normal];assert len(ns)==207 and len(set(ns))==207
 hover=frac(.012);terminal=frac(Y[ns[-1],47]);total=hover-terminal;guard=2*frac(ROOT_TOL)+2*frac(GAP_ROUND)
 ix=np.where(a['groups']=='mesh_normal_descent')[0];assert len(ix)==206
 # Exact saved row constants retain every existing original/inward buffer.
 # Original five normal and order rows bound each cardinal interval too.
 M=sparse.load_npz(V/'all_rows_matrix.npz');oldn=np.where(a['groups']=='old_normal_descent')[0];oldo=np.where(a['groups']=='old_approach_order')[0][-5:]
 labels=[1451,1452,1453,1454,1455,1456];ct=[times[i] for i in labels]
 lower=[];upper=[];groups=[]
 for e,(t0,t1) in enumerate(zip(nt[:-1],nt[1:])):
  b0,b1=ns[e:e+2];row=M.getrow(ix[e]);assert set(row.indices)=={int(b0)*75+71,int(b1)*75+71}
  h=next(i for i in range(5) if ct[i]<=t0 and t1<=ct[i+1]);ratio=(t1-t0)/(ct[h+1]-ct[h]);groups.append(h)
  u=min(frac(a['constant'][ix[e]]),frac(a['constant'][oldn[h]])*ratio)-guard
  l=max(guard,(-frac(a['constant'][oldo[h]])+guard)*ratio)
  lower.append(l);upper.append(u);assert 0<l<u
 # The first posthover state retains its full added posthover-gap reserve.
 first_post=np.where((a['groups']=='mesh_posthover') & (np.asarray(M[:,int(ns[1])*75+71].toarray()).ravel()!=0))[0]
 if len(first_post):lower[0]=max(lower[0],hover-min(frac(a['constant'][i]) for i in first_post)+frac(ROOT_TOL)+frac(GAP_ROUND))
 ref=[frac(Y[b0,47])-frac(Y[b1,47]) for b0,b1 in zip(ns[:-1],ns[1:])]
 drops,lam,breakindex=project_box_sum(ref,lower,upper,total)
 target=[hover];
 for d in drops:target.append(target[-1]-d)
 assert target[-1]==terminal
 gaps=np.array([float(t) for t in target]);target_control=np.c_[Y[:,47],X[:,3],X[:,16:24]];target_control[ns,0]=gaps
 # Gap-control basis: unconstrained prehover gaps and p; conserved loads;
 # normal-drop directions use one fixed pivot and cumulative incidence.
 ri=[];ci=[];vv=[];columns=[]
 def column(terms,meta):
  c=len(columns);columns.append(meta)
  for b,j,v in terms:
   if v:ri.append(int(b)*10+j);ci.append(c);vv.append(v)
 normal_set=set(ns.tolist())
 for b in range(K):
  if mesh['fixed'][b]:continue
  if b not in normal_set:column([(b,0,1.)],dict(kind='gap',state=b))
  column([(b,1,1.)],dict(kind='orientation',state=b))
  active=np.where(~mesh['inactive'][b])[0];lo=np.maximum(cm['lower'][b,2:],cm['reference'][b,2:]-.02);hi=np.minimum(cm['upper'][b,2:],cm['reference'][b,2:]+.02)
  pivot=int(active[np.argmax(np.minimum(X[b,16+active]-lo[active],hi[active]-X[b,16+active]))])
  for j in active:
   if j!=pivot:column([(b,2+int(j),1.),(b,2+pivot,-1.)],dict(kind='load',state=b,index=int(j),pivot=pivot))
 pivot=max(range(206),key=lambda e:(min(drops[e]-lower[e],upper[e]-drops[e]),-e))
 for e in range(206):
  if e==pivot:continue
  terms=[]
  for k,b in enumerate(ns[1:-1],1):
   v=-(int(e<k)-int(pivot<k))
   if v:terms.append((int(b),0,float(v)))
  column(terms,dict(kind='normal_drop',edge=e,pivot=pivot))
 B=sparse.coo_matrix((vv,(ri,ci)),shape=(K*10,len(columns))).tocsr();B.eliminate_zeros()
 T,G,A,rhs,raw=implicit(J);GB=sparse.block_diag(list(G),format='csr')@B
 allJ=(sparse.diags(1/a['scale'])@M@GB).tocsr();E=sparse.load_npz(Q/'equality_matrix.npz');eqJ=(E@GB).tocsr();eqJ.eliminate_zeros();assert eqJ.nnz==0
 # Every old +/- .02 f/p/load box becomes an explicit feature constraint,
 # including DEPENDENT f, rather than being lost when f is no longer a control.
 free=cm['free'];oldctrl=np.array([2,3]+list(range(16,24)));rr=np.arange(len(free));fc=free[:,0]*75+oldctrl[free[:,1]]
 C=sparse.coo_matrix((np.ones(len(free)),(rr,fc)),shape=(len(free),K*75)).tocsr();CJ=C@GB
 sparse.save_npz(O/'coordinate_basis.npz',B);sparse.save_npz(O/'feature_map.npz',GB);sparse.save_npz(O/'all_rows_gap_chain.npz',allJ);sparse.save_npz(O/'equalities_gap_chain.npz',eqJ);sparse.save_npz(O/'original_control_box_matrix.npz',C);sparse.save_npz(O/'original_control_box_chain.npz',CJ)
 np.savez_compressed(O/'implicit_map.npz',T=T,G=G,A=A,rhs=rhs,raw_output_chain=raw,referenceX=X,referenceY=Y,referenceJ=J,target_controls=target_control,normal_states=ns,original_control_lower=np.maximum(cm['lower'][tuple(free.T)],cm['reference'][tuple(free.T)]-.02),original_control_upper=np.minimum(cm['upper'][tuple(free.T)],cm['reference'][tuple(free.T)]+.02))
 data=dict(scope='New free normal-gap increment domain; not a complete path or forced linear-gap profile',normal_states=ns.tolist(),times=[pair(t) for t in nt],reference_drops=[pair(x) for x in ref],lower=[pair(x) for x in lower],upper=[pair(x) for x in upper],drops=[pair(x) for x in drops],target_gaps=[pair(x) for x in target],total=pair(total),lambda_value=pair(lam),break_index=breakindex,pivot=pivot,cardinal_groups=groups,mesh_rows=ix.tolist(),old_normal_rows=oldn.tolist(),old_order_rows=oldo.tolist(),root_tolerance=ROOT_TOL,rounding_half_ulp_bound=GAP_ROUND,edge_guard=pair(guard),columns=columns)
 (O/'layout.json').write_text(json.dumps(data))
 # Six genuinely new g/p/conserved-load JOINT derivative directions. No
 # completed440 full-root loop or old finite-work/AD fixture is repeated.
 s=load_inputs();selected_labels=[93,238,303,1450,610,657];blocks=[int(ls[i]) for i in selected_labels];oracle=Mechanics(s,mesh['canonical'][blocks]);xb=X[blocks].copy();gb=Y[blocks,47].copy();direction=np.zeros((6,10));direction[:,0]=.001;direction[:,1]=.02
 for j,b in enumerate(blocks):
  act=np.where(~mesh['inactive'][b])[0];direction[j,2+act[0]]=.01;direction[j,2+act[-1]]=-.01
 h=2e-5;outs=[];journals=[]
 for sign in [-1,1]:
  q=xb.copy();q[:,IND]+=sign*h*direction[:,1:];gap=gb+sign*h*direction[:,0];q[:,DEP]+=sign*h*np.einsum('nij,nj->ni',T[blocks][:,DEP,:],direction)
  xx,yy,journal=root(oracle,q,gap);outs.append(xx);journals.append(journal)
  np.savez_compressed(O/('root_minus.npz' if sign<0 else 'root_plus.npz'),X=xx,Y=yy,target_gap=gap)
  (O/'root_fixture_journal.json').write_text(json.dumps(journals))
 analytic=np.einsum('nij,nj->ni',T[blocks],direction);fd=(outs[1]-outs[0])/(2*h);error=float(abs(fd-analytic).max());assert error<2e-6,error
 np.savez_compressed(O/'root_direction_fixture.npz',blocks=blocks,direction=direction,h=h,analytic=analytic,finite_difference=fd)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,states=K,labels=len(ls),normal_edges=206,reduced_coordinates=B.shape[1],full_inequalities=len(a['constant']),retained_inequalities=int(a['keep'].sum()),original_equalities=E.shape[0],equality_chain_nonzeros=eqJ.nnz,original_control_bounds=len(free),maximum_condition=float(np.linalg.cond(A).max()),implicit_tangent_error=float(abs(A@T[:,DEP,:]-rhs).max()),gap_identity_tangent_error=float(abs(raw[:,47,:]-np.tile(np.r_[1.,np.zeros(9)],(K,1))).max()),normal_projection_maximum_change_m=max(abs(float(d-r)) for d,r in zip(drops,ref)),normal_projection_sum_exact=True,total_drop_m=float(total),sum_lower_m=float(sum(lower)),sum_upper_m=float(sum(upper)),root_fixture_states=6,root_fixture_signs=2,root_directional_error=error,unchanged_derivative_bound=2e-6,new_full_path_roots=0,new_trajectory_optimizers=0,new_actual_filter_ticks=0,new_physics_ticks=0,new_policy_updates=0,physical_admission=False,tangent_reference='Immutable440 X/Y/J; projection supplies new gap target coordinates only. Future complete lifting must recompute mechanics/J at the actual new state; this saved tangent is not claimed to be evaluated at projected targets.',limitations='Only new target-gap coordinates and local implicit tangent prepared. f remains dependent and its monotonicity/old +/- .02 bounds remain explicit hard constraints. No complete new candidate is reconstructed or admitted. Old geometry/support/static/command/rate/window rows all remain; no solver designed/frozen/launched.')
 (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
