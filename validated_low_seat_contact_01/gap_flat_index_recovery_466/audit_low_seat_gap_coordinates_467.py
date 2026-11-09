"""Independent SAVED-ARRAY gap-coordinate preparation audit; no root/FK/AD/optimizer."""
from pathlib import Path
from fractions import Fraction as F
import json,datetime,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'gap_coordinate_preparation_v2_466';O=D/'gap_coordinate_audit_467';read=lambda p:json.loads(Path(p).read_text());ff=lambda x:F.from_float(float(x))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def maximum(A):return float(abs(A.data).max(initial=0.))
def main():
 assert read(D/'gap_preparation_exit_466.json')['actual_exit']==0
 for n,h in read(P/'SHA256.json').items():assert sha(P/n)==h,n
 O.mkdir(exist_ok=False);d=read(P/'layout.json');v=dict(np.load(P/'implicit_map.npz'));r=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'));mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));cm=dict(np.load(D/'clock_row_preparation_446/coordinate_map.npz'))
 M=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz');E=sparse.load_npz(D/'clock_row_preparation_446/equality_matrix.npz');B=sparse.load_npz(P/'coordinate_basis.npz');K=len(v['referenceX']);dep=[0,1,2]+list(range(4,16));ind=[3]+list(range(16,24));eq=list(range(14))+[47]
 J=v['referenceJ'];T=v['T'];G=v['G'];assert np.array_equal(J,mesh['canonical_J']);assert np.array_equal(v['referenceX'],r['features'][:,:24]);assert np.array_equal(v['referenceY'],r['features'][:,24:])
 # Independently verify every defining15-equation tangent, then independently
 # assemble the exact analytic identity output rows; never threshold any row.
 JT=np.stack([J[b]@T[b] for b in range(K)]);target=np.zeros((K,15,10));target[:,-1,0]=1.
 tangent_error=float(abs(JT[:,eq,:]-target).max());assert tangent_error<2e-12
 assert np.array_equal(T[:,ind,1:],np.broadcast_to(np.eye(9),(K,9,9)));assert np.array_equal(T[:,ind,0],np.zeros((K,9)))
 rebuilt=np.concatenate([T,JT],axis=1);rebuilt[:,24:38,:]=0.;rebuilt[:,71,:]=0.;rebuilt[:,71,0]=1.
 chain_round=float(abs(rebuilt-G).max());assert chain_round<2e-12
 # Verify exact bounded-simplex KKT, endpoint and all original/dense caps.
 L=[F(*x) for x in d['lower']];U=[F(*x) for x in d['upper']];q=[F(*x) for x in d['drops']];ref=[F(*x) for x in d['reference_drops']];lam=F(*d['lambda_value']);total=F(*d['total']);gaps=[F(*x) for x in d['target_gaps']];ts=[F(*x) for x in d['times']];ns=d['normal_states'];guard=F(*d['edge_guard']);assert sum(q)==total and all(l<=x<=u for l,x,u in zip(L,q,U))
 for a,x,l,u in zip(ref,q,L,U):
  if l<x<u:assert a-x==lam
  elif x==l:assert a-x<=lam
  elif x==u:assert a-x>=lam
 for k in range(206):assert gaps[k]-gaps[k+1]==q[k]
 assert gaps[0]==ff(.012) and gaps[-1]==ff(v['referenceY'][ns[-1],47])
 assert guard==2*ff(1e-13)+2*F(1,2**59)
 for k,i in enumerate(d['mesh_rows']):assert U[k]+guard<=ff(r['constant'][i])
 for h,i in enumerate(d['old_normal_rows']):assert sum(q[k] for k in range(206) if d['cardinal_groups'][k]==h)+guard<=ff(r['constant'][i])
 for h,i in enumerate(d['old_order_rows']):assert sum(q[k] for k in range(206) if d['cardinal_groups'][k]==h)>=-ff(r['constant'][i])+guard
 # Exact rational values ensure at most half-ulp decoding error; fixed final
 # target matches the unchanged endpoint, not a new projected physical pose.
 assert np.array_equal(v['target_controls'][ns,0],np.array([float(x) for x in gaps]))
 max_round=max(abs(ff(v['target_controls'][b,0])-x) for b,x in zip(ns,gaps));assert max_round<=F(1,2**59)
 # Complete sparse incidence is reconstructed from independently checked
 # phase/load/state identities, with every dependent pivot retained.
 ri=[];ci=[];vv=[];seen=set()
 for c,meta in enumerate(d['columns']):
  kind=meta['kind'];terms=[]
  if kind in ('gap','orientation'):
   b=meta['state'];assert not mesh['fixed'][b]
   if kind=='gap':assert b not in ns
   terms=[(b,0 if kind=='gap' else 1,1.)]
  elif kind=='load':
   b,j,p=meta['state'],meta['index'],meta['pivot'];active=np.where(~mesh['inactive'][b])[0];lo=np.maximum(cm['lower'][b,2:],cm['reference'][b,2:]-.02);hi=np.minimum(cm['upper'][b,2:],cm['reference'][b,2:]+.02);expected=int(active[np.argmax(np.minimum(v['referenceX'][b,16+active]-lo[active],hi[active]-v['referenceX'][b,16+active]))]);assert p==expected and j in active and j!=p;terms=[(b,2+j,1.),(b,2+p,-1.)]
  else:
   assert kind=='normal_drop';e,p=meta['edge'],meta['pivot'];assert p==d['pivot'] and e!=p
   terms=[(b,0,float(int(p<k)-int(e<k))) for k,b in enumerate(ns[1:-1],1) if int(p<k)!=int(e<k)]
  key=tuple(sorted(meta.items()));assert key not in seen;seen.add(key)
  for b,j,x in terms:ri.append(b*10+j);ci.append(c);vv.append(x)
 BB=sparse.coo_matrix((vv,(ri,ci)),shape=B.shape).tocsr();assert maximum(B-BB)==0
 assert B.shape[1]==10540
 # All original rows remain in these products; local factorization validation
 # above is separate from this exact sparse arithmetic replay.
 Fmap=sparse.block_diag(list(G),format='csr')@BB;full=sparse.diags(1/r['scale'])@M@Fmap;equal=E@Fmap
 full_error=maximum(full-sparse.load_npz(P/'all_rows_gap_chain.npz'));eq_error=maximum(equal-sparse.load_npz(P/'equalities_gap_chain.npz'));assert full_error==eq_error==0
 equal.eliminate_zeros();assert equal.nnz==0
 free=cm['free'];ctrl=[2,3]+list(range(16,24));C=sparse.coo_matrix((np.ones(len(free)),(np.arange(len(free)),[int(b)*75+ctrl[int(j)] for b,j in free])),shape=(len(free),K*75)).tocsr();assert maximum(C-sparse.load_npz(P/'original_control_box_matrix.npz'))==0;assert maximum(C@Fmap-sparse.load_npz(P/'original_control_box_chain.npz'))==0
 # Replay ONLY saved new two-sign root fixtures, no model/root work here.
 fixture=dict(np.load(P/'root_direction_fixture.npz'));minus=dict(np.load(P/'root_minus.npz'));plus=dict(np.load(P/'root_plus.npz'));h=float(fixture['h']);blocks=fixture['blocks'];direction=fixture['direction'];fd=(plus['X']-minus['X'])/(2*h);analytic=np.stack([T[b]@direction[k] for k,b in enumerate(blocks)]);derivative=float(abs(fd-analytic).max());assert derivative<2e-6
 for sign,saved in [(-1,minus),(1,plus)]:
  assert np.array_equal(saved['X'][:,ind],v['referenceX'][blocks][:,ind]+sign*h*direction[:,1:]);assert abs(saved['Y'][:,:14]).max()<=1e-13;assert abs(saved['Y'][:,47]-saved['target_gap']).max()<=1e-13
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,arithmetic_only=True,normal_drop_KKT_exact=True,endpoint_sum_exact=True,all_206_dense_and5_original_normal_caps_preserved=True,original_order_reserves_preserved=True,root_error_guard_retained=True,maximum_gap_rounding_error_m=float(max_round),basis_error=maximum(B-BB),implicit_equation_error=tangent_error,local_output_chain_error=chain_round,full_row_chain_error=full_error,equality_chain_error=eq_error,equality_chain_nonzeros=equal.nnz,root_directional_error=derivative,derivative_bound=2e-6,all11906_original_control_bounds_including_dependent_f_retained=True,full_inequalities=331888,normal_drop_edges=206,reduced_coordinates=10540,new_root_calls=0,new_model_calls=0,new_optimizer_calls=0,physical_admission=False,complete_candidate=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
