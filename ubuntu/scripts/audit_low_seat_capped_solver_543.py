"""Independent saved-input/last-PDHG-transition and original-KKT audit. No optimizer or production solver import."""
from pathlib import Path
from fractions import Fraction as F
import json,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def ff(v):return F(float(v))
def same(A,B):
 A=A.tocsr();B=B.tocsr();assert A.shape==B.shape and np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)
def audit_solver(T):
 r=read(T/'result.json');assert r['full_size_robot_LP'] and not r['small_test'] and r['actual_lp_call'] and r['input_loading_inside_clock']
 assert r['configured_address_space_bytes']==40*1024**3 and not r['peak_memory_measured'];Q=T.parent;A=sparse.load_npz(Q/'matrix.npz');a=dict(np.load(Q/'arrays.npz'));E=sparse.load_npz(Q/'equality.npz');assert E.nnz==0 and np.all(a['equality_rhs']==0)
 if not (T/'input_proof.json').exists():
  assert not r['candidate'] and not (T/'solve/witness.npy').exists()
  return dict(passed=True,input_preparation_completed=False,complete_PDHG_transitions_verified=0,numerical_candidate=False,physical_admission=False)
 proof=read(T/'input_proof.json');assert proof['passed'];frozen=read(D/'capped_path_launch_542.json')['source_input_sha256']
 for p,h in {**proof['input_sha256'],**proof['full_input_sha256']}.items():assert sha(p)==h
 if proof['inherited']:
  for p,h in proof['input_sha256'].items():assert h==frozen[p]
  for n,h in proof['proof_sha256'].items():assert sha(D/n/'SHA256.json')==h and read(D/n/'independent_audit.json')['passed']
  base=D/'lifted_feasibility_preparation_490'/proof['role'];same(A,sparse.load_npz(base/'matrix.npz'));same(E,sparse.load_npz(base/'equality.npz'));old=dict(np.load(base/'arrays.npz'));assert all(np.array_equal(a[k],old[k]) for k in a)
 else:
  assert np.all(np.load(T/'new_implication_verified.npy'));met=np.load(T/'new_metric_verified.npz');assert met['rows'].all() and met['columns'].all()
 src=Path(proof['source']);SA=sparse.load_npz(src/'matrix.npz');sa=dict(np.load(src/'arrays.npz'));ids=np.asarray(proof['source_row_ids']);same(SA,A[ids]);assert np.array_equal(sa['rhs'],a['rhs'][ids])
 for k in ['cost','bounds','equality_rhs']:assert np.array_equal(sa[k],a[k])
 form=Path(proof['formulation']);H=sparse.load_npz(form/'hard.npz');S=sparse.load_npz(form/'coupled.npz');z=dict(np.load(form/'arrays.npz'));metric=dict(np.load(proof['metric']));bH=z['hard_rhs'];bS=z['coupled_rhs'];L,U,e=map(float,[z['lower_v'],z['upper_v'],z['objective_sign']]);sH=metric['sigma'][:len(bH)];sS=metric['sigma'][len(bH):];tau=metric['tau']
 out=dict(passed=True,input_preparation_completed=True,source_and_metric_proof_inherited=proof['inherited'],old_metric_split_or_implication_proofs_repeated=False,complete_PDHG_transitions_verified=0,numerical_candidate=bool(r['candidate']),physical_admission=False)
 if not (T/'solve/result.json').exists():assert not r['candidate'];return out
 records=read(T/'solve/iteration_metrics.json');assert len(records)==r['iterations']<=4096
 assert len(records)<=r['weighted_proximal_evaluations']<=r['weighted_proximal_attempts']<=len(records)+1 and len(records)<=r['native_updates']<=len(records)+1
 for k,rec in enumerate(records):
  assert rec['k']==k and max(rec['proximal']['orthant_relative'],rec['proximal']['hinge_relative'])<=1e-12
  assert all(np.isfinite(v) for v in rec['metrics'].values())
 out['saved_scalar_records_verified']=len(records)
 if records:
  t=dict(np.load(T/'solve/last_step.npz'));assert int(t['k'])==len(records)-1;err=max(float(np.max(abs(v),initial=0)) for v in [t['wH']-(t['old_y']+sH*(H@t['old_xbar'])),t['wS']-(t['old_lambda']+sS*(S@t['old_xbar'])),t['y']-np.maximum(t['wH']-sH*bH,0),t['x']-(t['old_x']-tau*(H.T@t['y']+S.T@t['lam'])),t['xbar']-(2*t['x']-t['old_x'])]);assert err<=1e-13
  if len(records)==1:assert all(np.all(t[k]==0) for k in ['old_x','old_xbar','old_y','old_lambda'])
  lam=list(map(ff,t['lam']));threshold=ff(t['threshold']);width=ff(U)-ff(L);mass=sum(lam,F(0));assert 0<=threshold<=width and all(v>=0 for v in lam)
  gap=width*max(F(0),mass-1)-threshold*(mass-1)
  for li,wi,si,bi in zip(lam,t['wS'],sS,bS):
   wi,si,bi=ff(wi),ff(si),ff(bi);q=wi-si*(bi+ff(L)+threshold);target=max(q,F(0));gap+=((li-target)**2+2*max(-q,F(0))*li)/(2*si)
  assert gap>=0
  out.update(complete_PDHG_transitions_verified=1,last_transition_index=len(records)-1,transition_association_error=err,last_saved_exact_proximal_gap=str(gap),last_saved_exact_proximal_gap_float=float(gap),all_intermediate_vectors_available=False,proximal_gap_scope='ONE saved last completed dual proximal objective gap; not uniform/summable error certificate or robot feasibility.')
 if r['candidate']:
  assert r['status']==0 and r['elapsed_s']<=45;can=dict(np.load(T/'solve/candidate.npz'));w=can['x'];assert np.array_equal(w,np.load(T/'solve/witness.npy')) and np.array_equal(w[:-1],t['x']) and np.array_equal(can['hard_dual'],t['y']) and np.array_equal(can['coupled_dual'],t['lam'])
  exact=ff(L);owner=-1;vx=list(map(ff,w[:-1]))
  for i in range(S.shape[0]):
   st,en=S.indptr[i:i+2];v=sum((ff(c)*vx[int(j)] for j,c in zip(S.indices[st:en],S.data[st:en])),F(0))-ff(bS[i])
   if v>exact:exact=v;owner=i
  sr=read(T/'solve/scalar_reconstruction.json');assert F(sr['exact_canonical_scalar'])==exact and sr['coupled_owner']==owner and e*w[-1]==float(exact) and F(sr['rounding_error'])==ff(e*w[-1])-exact
  sy=np.zeros(len(sa['rhs']));sy[z['hard_source_positions']]=can['hard_dual'];sy[z['coupled_source_positions']]=can['coupled_dual'];fy=np.zeros(len(a['rhs']));fy[ids]=sy;tu=float(can['upper_multiplier']);tl=float(can['lower_multiplier']);saved=np.load(T/'original_duals.npz');assert np.array_equal(saved['source'],sy) and np.array_equal(saved['full'],fy)
  checks={};res=np.load(T/'original_residuals.npz')
  for key,B,b,y in [('source',SA,sa,sy),('full',A,a,fy)]:
   primal=B@w-b['rhs'];station=b['cost']+B.T@y;station[-1]+=tu-tl;vi=max(0.,float(primal.max()),float(abs(E@w-a['equality_rhs']).max()),float((b['bounds'][:,0]-w).max()),float((w-b['bounds'][:,1]).max()));du=float(abs(station).max());gap=abs(float(b['cost']@w+b['rhs']@y+b['bounds'][-1,1]*tu-b['bounds'][-1,0]*tl))
   assert np.array_equal(res[key],primal) and np.array_equal(res[key+'_stationarity'],station);assert all(np.all(np.isfinite(v)) for v in [w,y,primal,station,tu,tl]) and np.all(y>=0) and tu>=0 and tl>=0 and max(vi,du,gap)<=1e-9;checks[key]=dict(primal=vi,dual=du,gap=gap)
  out.update(original_KKT=checks,once_rounded_scalar=sr,native_values_bitwise_unchanged=True)
 else:assert not (T/'solve/witness.npy').exists()
 return out
