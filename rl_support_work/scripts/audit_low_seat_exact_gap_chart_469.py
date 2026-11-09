"""Independent exact-arithmetic audit of new finite normal-drop chart468 only."""
from pathlib import Path
from fractions import Fraction as F
import math,json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'exact_gap_chart_preparation_468';Q=D/'gap_coordinate_preparation_v3_466';O=D/'exact_gap_chart_audit_469'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def fq(v):return F.from_float(float(v))
def rat(x):return F(int(x[0]),int(x[1]))
def floorq(q):
 v=float(q)
 if fq(v)>q:v=math.nextafter(v,-math.inf)
 return fq(v)
def ceilq(q):
 v=float(q)
 if fq(v)<q:v=math.nextafter(v,math.inf)
 return fq(v)
def diff(a,b):
 c=(a-b).tocsr();c.eliminate_zeros();return 0. if c.nnz==0 else float(abs(c.data).max())
def main():
 assert json.loads((D/'exact_gap_chart_exit_468.json').read_text())['actual_exit']==0
 for n,h in json.loads((P/'SHA256.json').read_text()).items():assert sha(P/n)==h,n
 O.mkdir(exist_ok=False);d=json.loads((P/'layout.json').read_text());old=json.loads((Q/'layout.json').read_text());lo=list(map(rat,d['lower']));hi=list(map(rat,d['upper']));total=rat(d['total']);p=d['pivot'];ids=d['independent'];U=F(1,2**59);assert p==old['pivot'] and ids==[i for i in range(206) if i!=p]
 for i,(l,u) in enumerate(zip(map(rat,old['lower']),map(rat,old['upper']))):
  assert lo[i]==(l if i==p else ceilq(l));assert hi[i]==(u if i==p else floorq(u))
 assert hi[p]-lo[p]>=U;assert sum(lo)<total<sum(hi)
 initial=list(map(rat,d['initial']));assert sum(initial)==total and all(l<=v<=u for l,v,u in zip(lo,initial,hi));oldinit=list(map(rat,old['drops']))
 for i in ids:
  cand=[x for x in [floorq(oldinit[i]),ceilq(oldinit[i])] if lo[i]<=x<=hi[i]];assert initial[i]==min(cand,key=lambda x:(abs(x-oldinit[i]),x))
 assert initial[p]==total-sum(initial[i] for i in ids)
 maps=0;largest=F(0);boundchecks=0
 def verify(origin,t,saved):
  nonlocal maps,largest,boundchecks
  ds=[F(0)]*206
  for i,v in zip(ids,t):ds[i]=fq(v)
  ds[p]=-sum(ds);rho=max([abs(v)/((hi[i]-origin[i]) if v>0 else (origin[i]-lo[i])) for i,v in enumerate(ds) if v] or [F(0)])
  alpha=1/(1+rho);q=[c+alpha*v for c,v in zip(origin,ds)];assert sum(q)==total
  z=[F(0)]*206
  for i in ids:z[i]=floorq(q[i])
  S=sum(z);lower=total-hi[p];upper=total-lo[p];raised=[]
  for i in ids:
   if S>=lower:break
   v=ceilq(q[i]);step=v-z[i];assert step<=U;z[i]=v;S+=step;raised.append(i)
  assert lower<=S<=upper;z[p]=total-S
  expected=list(map(rat,saved));assert z==expected;assert sum(z)==total and all(l<=v<=u for l,v,u in zip(lo,z,hi))
  assert all(abs(z[i]-q[i])<=U for i in ids) and abs(z[p]-q[p])<=205*U
  largest=max(largest,max(abs(a-b) for a,b in zip(z,q)));maps+=1
  # Conditional on each future actual gap root residual<=1e-13, the saved
  # original/prior/buffered dense and cardinal normal caps remain satisfied.
  exact=[rat(d['hover'])]
  for v in z:exact.append(exact[-1]-v)
  assert exact[-1]==rat(d['terminal']);rounded=list(map(lambda x:fq(float(x)),exact));assert all(abs(a-b)<=U for a,b in zip(exact,rounded))
  guard=rat(old['edge_guard']);tol=F.from_float(1e-13)
  for k,(a,b) in enumerate(zip(rounded[:-1],rounded[1:])):
   assert a-b+2*tol<=rat(old['upper'][k])+guard
   assert a-b-2*tol>=rat(old['lower'][k])-guard
  boundchecks+=206
  return alpha,rho,raised
 center=list(map(rat,d['fixture_center']));a=(total-sum(lo))/(sum(hi)-sum(lo));cq=[l+a*(u-l) for l,u in zip(lo,hi)];assert sum(center)==total and all(l<v<u for l,v,u in zip(lo,center,hi))
 fixtures=json.loads((P/'ray_fixtures.json').read_text());assert len(fixtures)==205;maxerror=0
 for k,f in enumerate(fixtures):
  assert f['edge']==ids[k];h=f['h'];eps=f['eps'];rows=[]
  for v,saved in zip([h-eps,h,h+eps],f['outputs']):
   t=np.zeros(205);t[k]=v;alpha,rho,raised=verify(center,t,saved);rows.append(list(map(rat,saved)))
   if v==h:assert alpha==rat(f['alpha']) and rho==rat(f['rho']) and raised==f['raised']
  fd=np.array([float((b-a)/(fq(h+eps)-fq(h-eps))) for a,b in zip(rows[0],rows[2])]);der=np.zeros(206);v=float(1/(1+rat(f['rho']))**2);der[ids[k]]=v;der[p]=-v;err=float(abs(fd-der).max());maxerror=max(maxerror,err);assert err==f['error'] and err<2e-6
 previous=center
 for f in json.loads((P/'joint_chain.json').read_text()):
  origin=list(map(rat,f['origin']));assert origin==previous;alpha,rho,raised=verify(origin,f['tangent'],f['output']);assert alpha==rat(f['alpha']) and rho==rat(f['rho']) and raised==f['raised'];previous=list(map(rat,f['output']));exact=[rat(d['hover'])]
  for v in previous:exact.append(exact[-1]-v)
  assert exact==list(map(rat,f['exact_gaps']));assert np.array_equal(np.asarray([float(v) for v in exact]),np.asarray(f['gaps']))
 # Independently rebuild cumulative gap incidence for EVERY205 normal-drop
 # column, then chain with saved reference mechanics and ALL original rows.
 states=old['normal_states'];K=1410;rr=[];cc=[];vv=[]
 for j,e in enumerate(ids):
  for k,b in enumerate(states[1:-1],1):
   v=-(int(e<k)-int(p<k))
   if v:rr.append(b*10);cc.append(j);vv.append(v)
 B=sparse.csr_matrix((vv,(rr,cc)),shape=(K*10,205));v=dict(np.load(Q/'implicit_map.npz'));G=sparse.block_diag(list(v['G']),format='csr');a=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'));M=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz');A=(sparse.diags(1/a['scale'])@M@G@B).tocsr();C=sparse.load_npz(Q/'original_control_box_matrix.npz')@G@B
 ae=diff(A,sparse.load_npz(P/'normal_drop_full_chain.npz'));ce=diff(C,sparse.load_npz(P/'normal_drop_original_box_chain.npz'));assert max(ae,ce)==0
 E=sparse.load_npz(D/'clock_row_preparation_446/equality_matrix.npz')@G@B;E.eliminate_zeros();assert E.nnz==0
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,replayed_finite_maps=maps,ray_fixtures=205,joint_chain_maps=3,exact_sum_every_map=True,dependent_pivot_bounds_checked=True,independent_inward_endpoint_checks=True,conditional_dense_root_error_checks=boundchecks,derivative_error=maxerror,unchanged_bound=2e-6,maximum_rounding_defect_m=float(largest),full331888_reference_row_chain_error=ae,all11906_original_box_chain_error=ce,equality_chain_nonzeros=E.nnz,finite_map_is_quantized=True,no_claim_of_literal_stored_map_derivative=True,model_calls=0,root_calls=0,optimizer_calls=0,new_complete_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
