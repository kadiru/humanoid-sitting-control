"""Independent490 structural elimination, LP assembly and NEW finite-map audit."""
from pathlib import Path
from fractions import Fraction as F
import math,json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'lifted_feasibility_preparation_490';O=D/'lifted_feasibility_audit_491'
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def fq(x):return F.from_float(float(x))
def rat(x):return F(int(x[0]),int(x[1]))
def error(a,b):
 c=(a-b).tocsr();c.eliminate_zeros();return float(abs(c.data).max()) if c.nnz else 0.
def mul(A,t):
 q=list(map(fq,t));v=[]
 for i in range(A.shape[0]):v.append(sum((fq(A.data[j])*q[A.indices[j]] for j in range(A.indptr[i],A.indptr[i+1])),F(0)))
 return v
def floor(q):
 v=float(q);return math.nextafter(v,-math.inf) if fq(v)>q else v
def ceil(q):
 v=float(q);return math.nextafter(v,math.inf) if fq(v)<q else v

def main():
 assert read(D/'lifted_preparation_exit_490.json')['actual_exit']==0
 for n,h in read(P/'SHA256.json').items():assert sha(P/n)==h,n
 for n,h in read(D/'lifted_preparation_launch_490.json')['source_input_sha256'].items():assert sha(n)==h,n
 O.mkdir(exist_ok=False);z=dict(np.load(P/'reference.npz'));seed=dict(np.load(D/'free_gap_path_472/initial.npz'));mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));cm=dict(np.load(D/'clock_row_preparation_446/coordinate_map.npz'));row=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'));ea=dict(np.load(D/'clock_row_preparation_446/equality_arrays.npz'));active=np.flatnonzero(~mesh['fixed']);K=len(seed['X']);n0=10540
 output=np.r_[np.arange(14),np.arange(15,51)];extra=[0,1,2]+list(range(4,16))+list(range(39,71))+[72,73,74];oldextra=read(D/'feature_collocation_preparation_488/layout.json')['extra_features'];oldindices=[oldextra.index(v) for v in extra]
 selected=np.r_[np.arange(n0),(n0+np.arange(len(active))[:,None]*65+np.asarray(oldindices)).ravel()];oldP=sparse.load_npz(D/'feature_collocation_preparation_488/feature_basis.npz');S=oldP[:,selected].tocsr();assert S.shape==(105750,78790) and error(S,sparse.load_npz(P/'feature_basis.npz'))==0
 oldQ=sparse.load_npz(D/'feature_collocation_preparation_488/closure_jacobian.npz');selected_rows=(np.arange(len(active))[:,None]*65+output).ravel();v=.02*np.sum(abs(cm['G']),axis=2);sigma=np.einsum('nyx,nx->ny',abs(seed['J']),v[:,:24])+v[:,24:];assert np.array_equal(v,z['feature_scale']) and np.array_equal(sigma,z['sigma']);assert np.all(sigma[active][:,output]>0)
 C=oldQ[selected_rows][:,selected].multiply((1/sigma[active][:,output].ravel())[:,None]).tocsr();assert error(C,sparse.load_npz(P/'consistency_chain.npz'))==0
 assert np.all(np.load(D/'contact_lift_440/states.npz')['normal'][mesh['canonical']][active]==0)
 features=np.c_[seed['X'],seed['Y']];source=features.copy();features[active,24:38]=0.;features[active,71]=seed['target'][active,0]
 for b in active:features[b,38]=float(sum(map(fq,features[b,16:24]),F(0))-1)
 assert np.array_equal(features,z['features']) and np.array_equal(source,z['source_features']) and np.array_equal(seed['J'],z['J']) and np.array_equal(seed['Y'],z['actualY'])
 M=sparse.load_npz(D/'clock_row_preparation_v2_446/all_rows_matrix.npz');E=sparse.load_npz(D/'clock_row_preparation_446/equality_matrix.npz');A=(M@S).multiply((1/row['scale'])[:,None]).tocsr();assert error(A,sparse.load_npz(P/'all_rows_chain.npz'))==0
 r=(M@features.ravel()+row['constant'])/row['scale'];h=((features[active,24:][:,output]-seed['Y'][active][:,output])/sigma[active][:,output]).ravel();assert np.array_equal(r,z['r']) and np.array_equal(h,z['h']);assert np.array_equal(E@features.ravel()+ea['constant'],z['e']);assert (E@S).nnz==0
 free=cm['free'];ref=cm['reference'].ravel()[free];lo=np.maximum(cm['lower'].ravel()[free],ref-.02);hi=np.minimum(cm['upper'].ravel()[free],ref+.02);controlrows=(free//10)*75+np.array([2,3]+list(range(16,24)))[free%10];varying=np.flatnonzero(np.diff(S.indptr));assert np.array_equal(lo,z['lower']) and np.array_equal(hi,z['upper']) and np.array_equal(controlrows,z['control_feature_rows']) and np.array_equal(varying,z['varying'])
 gl=read(D/'gap_coordinate_preparation_v3_466/layout.json');dom=read(D/'exact_gap_chart_preparation_468/layout.json');low=list(map(rat,dom['lower']));high=list(map(rat,dom['upper']));total=rat(dom['total']);pivot=dom['pivot'];ind=dom['independent'];U=F(1,2**59);normalstates=np.asarray(gl['normal_states']);ngset=set((normalstates*75+71).tolist());rr=[];cc=[];vv=[]
 for j,c in enumerate(gl['columns']):
  if c['kind']=='normal_drop':rr += [c['edge'],c['pivot']];cc += [j,j];vv += [1.,-1.]
 N=sparse.csr_matrix((vv,(rr,cc)),shape=(206,S.shape[1]));assert error(N,sparse.load_npz(P/'normal_drop_chain.npz'))==0
 H=sparse.vstack([S[controlrows],S[varying],N],format='csr');assert error(H,sparse.load_npz(P/'interval_matrix.npz'))==0
 meta=read(P/'intervals.json');drops=list(map(rat,meta['drops']))
 def intervals_at(origin,drops):
  f=origin.ravel();lower=[];upper=[];cl=[];cu=[];owners=[]
  allrows=list(controlrows)+list(varying)
  for j,rowid in enumerate(allrows):
   if j<len(lo):l=float(lo[j]);u=float(hi[j]);owner=['original_control',j]
   else:l=float(f[rowid]-v.ravel()[rowid]);u=float(f[rowid]+v.ravel()[rowid]);owner=['feature_trust',int(rowid)]
   c=fq(f[rowid]);cl.append(fq(l)-c);cu.append(fq(u)-c);owners.append(owner)
   if j>=len(lo) and int(rowid) in ngset:lower.append(fq(l)-c+207*U);upper.append(fq(u)-c-207*U)
   else:lower.append(fq(math.nextafter(l,math.inf))-c);upper.append(fq(math.nextafter(u,-math.inf))-c)
  for i,(l,u,c) in enumerate(zip(low,high,drops)):
   err=(205 if i==pivot else 1)*U;cl.append(l-c);cu.append(u-c);lower.append(l-c+err);upper.append(u-c-err);owners.append(['drop',i])
  return lower,upper,[u-l for l,u in zip(lower,upper)],cl,cu,owners
 lower,upper,width,cl,cu,owners=intervals_at(features,drops)
 for name,expected in [('lower',lower),('upper',upper),('widths',width),('closed_lower',cl),('closed_upper',cu)]:assert list(map(rat,meta[name]))==expected,name
 assert meta['owners']==owners and all(w>0 for w in width)
 keep=row['keep'];soft=row['soft']&keep;hard=keep&~soft;variable=np.diff(A.indptr)>0;guard=np.zeros(len(r));guard[soft&variable]=2.**-20;hp=hard&variable;guard[hp]=np.where(r[hp]>0,np.minimum(2.**-20,.5*r[hp]),2.**-20);assert np.array_equal(guard,z['guard']);phi=max(0.,float(-r[soft].min()),float(abs(h).max()));HN=H.multiply((1/np.asarray([float(w) for w in width]))[:,None]).tocsr();n=S.shape[1];m=C.shape[0];closed=sparse.hstack([HN,sparse.csr_matrix((H.shape[0],1))]);closed_rhs=np.r_[[float(u/w) for u,w in zip(cu,width)],[-float(l/w) for l,w in zip(cl,width)]];assembly={}
 for mode in ['primary','witness']:
  ceiling=None if mode=='primary' else .5*phi
  if ceiling is None:
   top=sparse.hstack([-A[keep],sparse.csr_matrix(-soft[keep,None].astype(float))]);last=sparse.csr_matrix(-np.ones((m,1)));clos=sparse.vstack([sparse.hstack([C,last]),sparse.hstack([-C,last])]);rhs=np.r_[r[keep]-guard[keep],-h,h,closed_rhs];cost=np.r_[np.zeros(n),1.];bounds=np.c_[np.r_[np.full(n,-np.inf),0.],np.r_[np.full(n,np.inf),max(1.,2*phi)]]
  else:
   top=sparse.hstack([-A[keep],sparse.csr_matrix((keep.sum(),1))]);last=sparse.csr_matrix((m,1));clos=sparse.vstack([sparse.hstack([C,last]),sparse.hstack([-C,last])]);rhs=np.r_[r[keep]-guard[keep]+soft[keep]*ceiling,-h+ceiling,h+ceiling,closed_rhs];cost=np.r_[np.zeros(n),-1.];bounds=np.c_[np.r_[np.full(n,-np.inf),0.],np.r_[np.full(n,np.inf),.5]]
  parts=[top,clos,closed,-closed]
  if ceiling is not None:
   one=sparse.csr_matrix(np.ones((H.shape[0],1)));parts += [sparse.hstack([HN,one]),sparse.hstack([-HN,one])];rhs=np.r_[rhs,[float(u/w) for u,w in zip(upper,width)],[-float(l/w) for l,w in zip(lower,width)]]
  mat=sparse.vstack(parts,format='csr');p=P/mode;a=dict(np.load(p/'arrays.npz'));err=error(mat,sparse.load_npz(p/'matrix.npz'));assert err==0 and np.array_equal(rhs,a['rhs']) and np.array_equal(cost,a['cost']) and np.array_equal(bounds,a['bounds']);eq=sparse.load_npz(p/'equality.npz');assert eq.shape==(1366,n+1) and eq.nnz==0 and np.all(a['equality_rhs']==0);assembly[mode]=dict(shape=mat.shape,matrix_error=err,rhs_error=0,fixture_only=True)
 # Independent exact reconstruction of ONLY these NEW three full-coordinate
 # maps. No production decoder/joint_round or old fixture loop is called.
 fz=dict(np.load(P/'finite_joint_fixtures.npz'));fm=read(P/'finite_fixture_metadata.json');origin=fz['origin'];cd=list(map(rat,fm['drop_origin']));fl,fu,fw,fcl,fcu,_=intervals_at(origin,cd);checks=[]
 for t,saved,record in zip(fz['directions'],fz['outputs'],fm['nonzero']):
  vals=mul(H,t);eta=min([F(1,2)]+[min(v-l,u-v)/w for v,l,u,w in zip(vals,fl,fu,fw)]);assert eta>0 and rat(record['certificate']['eta'])==eta
  delta=mul(S,t);q=[fq(c)+v for c,v in zip(origin.ravel(),delta)];candidate=np.asarray(list(map(float,q))).reshape(origin.shape);dd=mul(N,t);qd=[c+v for c,v in zip(cd,dd)];assert sum(qd)==total and all(l<=x<=u for l,x,u in zip(low,qd,high))
  encoded=[F(0)]*206
  for i in ind:encoded[i]=fq(floor(qd[i]))
  mass=sum(encoded);raised=[]
  for i in ind:
   if mass>=total-high[pivot]:break
   new=fq(ceil(qd[i]));mass+=new-encoded[i];encoded[i]=new;raised.append(i)
  encoded[pivot]=total-mass;assert sum(encoded)==total and all(l<=x<=u for l,x,u in zip(low,encoded,high));assert encoded==list(map(rat,record['drop_state'])) and raised==record['raised']
  gap=[rat(dom['hover'])]
  for x in encoded:gap.append(gap[-1]-x)
  assert gap[-1]==rat(dom['terminal']);candidate[normalstates,71]=list(map(float,gap));candidate[active,24:38]=0.;worst=F(0)
  for b in active:
   before=sum(map(fq,origin[b,16:24]),F(0));after=sum(map(fq,candidate[b,16:24]),F(0));assert sum(q[b*75+16:b*75+24],F(0))==before;worst=max(worst,abs(after-before));candidate[b,38]=float(after-1)
  assert np.array_equal(candidate,saved);assert worst==rat(record['load_sum_rounding']) and worst<=8*F(1,2**54);assert record['shared_alpha']==[1,1]
  assert np.array_equal(candidate[mesh['fixed']],origin[mesh['fixed']]) and np.all(candidate[:,16:24][mesh['inactive']]==0)
  actualcontrol=candidate.ravel()[controlrows];assert np.all(lo<=actualcontrol) and np.all(actualcontrol<=hi)
  for i,j in enumerate(varying):actualdelta=fq(candidate.ravel()[j])-fq(origin.ravel()[j]);assert fcl[len(lo)+i]<=actualdelta<=fcu[len(lo)+i]
  checks.append(dict(exact_replay=True,eta=float(eta),load_sum_rounding=float(worst),drop_endpoint_exact=True))
 zero_eta=min([F(1,2)]+[min(-l,u)/w for l,u,w in zip(fl,fu,fw)]);assert zero_eta>0 and rat(fm['zero']['certificate']['eta'])==zero_eta
 # Zero leaves the canonical finite representation unchanged, not a new map.
 assert fm['zero']['drop_state']==fm['drop_origin'] and fm['zero']['load_sum_rounding']==[0,1]
 prep=read(P/'preparation.json');assert not prep['outward_certificate']['passed'] and prep['new_direct_coordinates_covered']==68250
 assert prep['gap_trust_encoding_error_bound_m']==float(207*U)
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,lifted_coordinates=n,consistency_rows=m,intervals=H.shape[0],structural_elimination_from488_error=0,canonical_reference_error=0,scales_error=0,consistency_matrix_error=0,assembly=assembly,new_finite_map_checks=checks,zero_identity_verified=True,normal_gap_trust_rounding_bound_m=float(207*U),all_original_rows_boxes_buffers_retained=True,reference_actualX_unchanged=True,auxiliary_success_is_not_actual_feasibility=True,model_calls=0,root_calls=0,ad_calls=0,lp_calls=0,new_complete_path=False,physical_admission=False)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
