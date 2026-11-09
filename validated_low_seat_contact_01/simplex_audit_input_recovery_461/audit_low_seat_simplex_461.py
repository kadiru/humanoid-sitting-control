"""Independent arithmetic/basis/row-chain audit, no mechanics or optimizer."""
from pathlib import Path
from fractions import Fraction as Q
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'simplex_preparation_460';O=D/'simplex_audit_461'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def replay(c,t,lo,hi,pivot):
 c=list(map(lambda x:Q(float(x)),c));lo=list(map(lambda x:Q(float(x)),lo));hi=list(map(lambda x:Q(float(x)),hi))
 delta=[];k=0
 for i in range(len(c)):
  if i==pivot:delta.append(Q(0))
  else:delta.append(Q(float(t[k])));k+=1
 if pivot>=0:delta[pivot]=-sum(delta)
 inverse_alpha=Q(1)
 for i,v in enumerate(delta):
  if v:
   room=hi[i]-c[i] if v>0 else c[i]-lo[i]
   assert room>0
   inverse_alpha=max(inverse_alpha,1+abs(v)/room)
 exact=[c[i]+delta[i]/inverse_alpha for i in range(len(c))]
 assert all(lo[i]<=exact[i]<=hi[i] for i in range(len(c)))
 if pivot>=0:assert sum(exact)==sum(c)
 result=np.array(list(map(float,exact)));assert all(float(lo[i])<=result[i]<=float(hi[i]) for i in range(len(c)))
 return result,float(1/inverse_alpha),np.array(list(map(float,delta)))
def main():
 O.mkdir(exist_ok=False)
 for name,h in json.loads((P/'SHA256.json').read_text()).items():assert sha(P/name)==h
 f=dict(np.load(P/'joint_fixture.npz'));groups=json.loads((P/'layout.json').read_text());prep=json.loads((P/'preparation.json').read_text());rays=json.loads((P/'ray_fixtures.json').read_text())
 c=f['origin'];lo=f['lower'];hi=f['upper'];cm=dict(np.load(D/'clock_row_preparation_446/coordinate_map.npz'));mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'))
 assert np.array_equal(c,cm['reference'].ravel()[cm['free']]);assert np.array_equal(c,f['zero'])
 assert np.array_equal(lo,np.maximum(cm['lower'].ravel()[cm['free']],c-.02));assert np.array_equal(hi,np.minimum(cm['upper'].ravel()[cm['free']],c+.02))
 free_pos={int(v):i for i,v in enumerate(cm['free'])};seen=[];column_owner=[];col=0;sum_errors=[];max_sum=0.;max_count=0
 for gi,g in enumerate(groups):
  ids=g['indices'];state=g['state'];slots=([0] if g['kind']=='phase_f' else [1] if g['kind']=='phase_p' else list(range(2,10)))
  assert ids==[free_pos[state*10+s] for s in slots if state*10+s in free_pos]
  pivot=g['pivot'];expected=-1 if g['kind']!='loads' else int(np.argmax(np.minimum(c[ids]-lo[ids],hi[ids]-c[ids])))
  assert pivot==expected;assert g['independent']==[i for i in range(len(ids)) if i!=pivot]
  assert g['columns']==list(range(col,col+len(g['independent'])));col+=len(g['columns']);seen+=ids
  for j in g['independent']:column_owner.append((ids[j],None if pivot<0 else ids[pivot]))
  v,a,_=replay(c[ids],f['tangent'][g['columns']],lo[ids],hi[ids],pivot)
  assert np.array_equal(v,f['mapped'][ids]) and a==f['alphas'][gi]
  if pivot>=0:
   delta=sum(map(lambda x:Q(float(x)),v))-sum(map(lambda x:Q(float(x)),c[ids]));sum_errors.append(float(delta));assert abs(delta)<=len(ids)*Q(1,2**54)
   weights=mesh['referenceX'][state,16:24];max_sum=max(max_sum,float(abs(sum(map(lambda x:Q(float(x)),weights))-1)));max_count=max(max_count,len(ids))
 assert sorted(seen)==list(range(len(c))) and len(seen)==len(c)
 assert np.array_equal(sum_errors,f['sum_errors'])
 rr=[];cc=[];vv=[]
 for j,(positive,negative) in enumerate(column_owner):
  rr.append(positive);cc.append(j);vv.append(1.)
  if negative is not None:rr.append(negative);cc.append(j);vv.append(-1.)
 basis=sparse.csr_matrix((vv,(rr,cc)),shape=(len(c),col));diff=basis-sparse.load_npz(P/'null_basis.npz');assert diff.nnz==0
 # Reconstruct each column directly from saved original columns; no mechanics,
 # root, AD, or multiplication by the preparation's saved basis is used.
 V=D/'clock_row_preparation_v2_446';scale=np.load(V/'row_arrays.npz')['scale']
 A=sparse.load_npz(V/'global_jacobian.npz').multiply((1/scale)[:,None]).tocsc();E=sparse.load_npz(D/'clock_row_preparation_446/equality_jacobian.npz').tocsc()
 rebuilt=sparse.hstack([A[:,p] if n is None else A[:,p]-A[:,n] for p,n in column_owner],format='csr')
 rebuilt_e=sparse.hstack([E[:,p] if n is None else E[:,p]-E[:,n] for p,n in column_owner],format='csr')
 dif=rebuilt-sparse.load_npz(P/'all_rows_null_chain.npz');dif.eliminate_zeros();assert dif.nnz==0
 dife=rebuilt_e-sparse.load_npz(P/'equalities_null_chain.npz');dife.eliminate_zeros();assert dife.nnz==0
 normal=rebuilt_e[:-1].copy();normal.eliminate_zeros();assert normal.nnz==0
 error=0.
 for r in rays:
  g=groups[r['group']];ids=g['indices'];local=g['columns'].index(r['column']);direction=np.zeros(len(g['columns']));direction[local]=1
  mid,a,delta=replay(c[ids],direction*r['amplitude'],lo[ids],hi[ids],g['pivot'])
  plus,_,_=replay(c[ids],direction*(r['amplitude']+r['h']),lo[ids],hi[ids],g['pivot']);minus,_,_=replay(c[ids],direction*(r['amplitude']-r['h']),lo[ids],hi[ids],g['pivot'])
  assert np.array_equal(mid,r['middle']) and np.array_equal(plus,r['plus']) and np.array_equal(minus,r['minus'])
  slope=delta/r['amplitude']*a*a;fd=(plus-minus)/(2*r['h']);error=max(error,float(abs(fd-slope).max()))
 assert len(rays)==col and error<2e-6
 u=2.**-53;bound=max_sum+120*max_count*2.**-54+(7*u/(1-7*u))*(1+max_sum+120*max_count*2.**-54)
 assert bound==prep['rounding_proof']['maximum_evaluated_normal_bound'] and bound<1e-13
 norm=np.load(D/'contact_lift_440/states.npz')['normal'][mesh['canonical']];assert np.all(norm[~mesh['fixed']]==0)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,arithmetic_only=True,coordinates=col,free_original_controls=len(c),load_groups=len(sum_errors),all_rational_outputs_exact_replay=True,null_basis_error=0.,all_rows_chain_error=0.,equality_chain_error=0.,normal_tangent_nonzeros=0,ray_derivative_error=error,derivative_bound=2e-6,maximum_evaluated_normal_bound=bound,normal_bound=1e-13,model_calls=0,root_calls=0,optimizer_calls=0,new_candidate=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
