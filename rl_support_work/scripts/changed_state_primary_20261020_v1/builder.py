"""New state-dependent C, residuals, guards, intervals and full490 primary."""
from common import *
from fractions import Fraction as F
import numpy as np
from scipy import sparse
U=F(1,2**59)
fq=lambda x:F.from_float(float(x))
OUTPUT=np.r_[np.arange(14),np.arange(15,51)]

def load():
 z=dict(np.load(BASE/'reference.npz'));f=np.load(PRIOR/'mapped.npz')['features']
 actual=dict(np.load(PRIOR/'actual.npz'));aj=dict(np.load(PRIOR/'accepted_J.npz'))
 y=actual['Y'];J=aj['J'];mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'))
 assert np.array_equal(actual['changed'],aj['changed']) and not np.array_equal(f,z['features'])
 assert np.array_equal(f[mesh['fixed']],z['features'][mesh['fixed']])
 assert read(PRIOR/'summary.json')['trial_merit_accepted'] and read(PRIOR/'audit.json')['passed']
 dom=read(D/'exact_gap_chart_preparation_468/layout.json')
 for k in ['lower','upper']:dom[k]=[F(*x) for x in dom[k]]
 gl=read(D/'gap_coordinate_preparation_v3_466/layout.json')
 d=dict(P=sparse.load_npz(BASE/'feature_basis.npz').tocsr(),N=sparse.load_npz(BASE/'normal_drop_chain.npz').tocsr(),
   rows=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz')),
   eq=dict(np.load(D/'clock_row_preparation_446/equality_arrays.npz')),
   active=z['active'],sigma=z['sigma'],feature_scale=z['feature_scale'],varying=z['varying'],
   control_feature_rows=z['control_feature_rows'],lo=z['lower'],hi=z['upper'],dom=dom,
   normal_gap_rows=set((np.asarray(gl['normal_states'])*75+71).tolist()))
 A=sparse.load_npz(STATE/'A.npz').tocsr();P=d['P'];active=d['active']
 L=np.zeros((len(active),50,75));L[:,:,:24]=-J[active][:,OUTPUT,:]
 for i,v in enumerate(OUTPUT):L[:,i,24+v]=1.
 ids=(active[:,None]*75+np.arange(75)).ravel()
 C=(sparse.block_diag(list(L),format='csr')@P[ids]).multiply((1/d['sigma'][active][:,OUTPUT].ravel())[:,None]).tocsr()
 r=actual['auxiliary_rows'];e=actual['auxiliary_equalities']
 h=(actual['consistency'][active][:,OUTPUT]/d['sigma'][active][:,OUTPUT]).ravel()
 soft=d['rows']['soft']&d['rows']['keep'];hard=d['rows']['keep']&~soft;variable=np.diff(A.indptr)>0
 guard=np.zeros(len(r));guard[soft&variable]=2.**-20;hp=hard&variable
 guard[hp]=np.where(r[hp]>0,np.minimum(2.**-20,.5*r[hp]),2.**-20)
 mu=read(PRIOR/'decision.json')['summary']['phi']
 assert mu==max(0.,float(-r[soft].min()),float(abs(h).max()))
 b=dict(A=A,C=C,r=r,e=e,h=h,guard=guard,phi=mu)
 drops=[F(*x) for x in read(PRIOR/'map.json')['drop_state']]
 return d,f,y,J,drops,b

def intervals(d,features,drops,trust):
 P=d['P'];ids=d['varying'];f=features.ravel();v=d['feature_scale'].ravel();H=sparse.vstack([P[d['control_feature_rows']],P[ids],d['N']],format='csr');lower=[];upper=[];cl=[];cu=[];owners=[]
 def add(l,u,c,owner,encoding=None):
  l=float(l);u=float(u);cl.append(fq(l)-c);cu.append(fq(u)-c)
  if encoding is None:lower.append(fq(np.nextafter(l,np.inf))-c);upper.append(fq(np.nextafter(u,-np.inf))-c)
  else:lower.append(fq(l)-c+encoding);upper.append(fq(u)-c-encoding)
  owners.append(owner)
 for i,row in enumerate(d['control_feature_rows']):add(d['lo'][i],d['hi'][i],fq(f[row]),['original_control',i])
 for row in ids:add(f[row]-trust*v[row],f[row]+trust*v[row],fq(f[row]),['feature_trust',int(row)],207*U if row in d['normal_gap_rows'] else None)
 for i,(c,l,u) in enumerate(zip(drops,d['dom']['lower'],d['dom']['upper'])):
  e=205*U if i==d['dom']['pivot'] else U;cl.append(l-c);cu.append(u-c);lower.append(l-c+e);upper.append(u-c-e);owners.append(['drop',i])
 widths=[u-l for l,u in zip(lower,upper)];assert all(w>0 for w in widths)
 return dict(H=H,lower=lower,upper=upper,widths=widths,closed_lower=cl,closed_upper=cu,owners=owners)

def assemble(d,b,s,ceiling=None):
 n=d['P'].shape[1];keep=d['rows']['keep'];soft=d['rows']['soft'][keep].astype(float);k=int(keep.sum());m=b['C'].shape[0];H=s['H'];width=np.array([float(x) for x in s['widths']]);HN=H.multiply((1/width)[:,None]).tocsr();closed_rhs=np.r_[[float(u/w) for u,w in zip(s['closed_upper'],s['widths'])],[-float(l/w) for l,w in zip(s['closed_lower'],s['widths'])]]
 if ceiling is None:
  top=sparse.hstack([-b['A'][keep],sparse.csr_matrix(-soft[:,None])]);cr=sparse.csr_matrix(-np.ones((m,1)));closure=sparse.vstack([sparse.hstack([b['C'],cr]),sparse.hstack([-b['C'],cr])]);rhs=np.r_[b['r'][keep]-b['guard'][keep],-b['h'],b['h'],closed_rhs];cost=np.r_[np.zeros(n),1.];last=[0.,max(1.,2*b['phi'])]
 else:
  top=sparse.hstack([-b['A'][keep],sparse.csr_matrix((k,1))]);closure=sparse.vstack([sparse.hstack([b['C'],sparse.csr_matrix((m,1))]),sparse.hstack([-b['C'],sparse.csr_matrix((m,1))])]);rhs=np.r_[b['r'][keep]-b['guard'][keep]+soft*ceiling,-b['h']+ceiling,b['h']+ceiling,closed_rhs];cost=np.r_[np.zeros(n),-1.];last=[0.,.5]
 closed=sparse.hstack([HN,sparse.csr_matrix((H.shape[0],1))]);parts=[top,closure,closed,-closed]
 if ceiling is not None:
  ones=sparse.csr_matrix(np.ones((H.shape[0],1)));parts += [sparse.hstack([HN,ones]),sparse.hstack([-HN,ones])];rhs=np.r_[rhs,[float(u/w) for u,w in zip(s['upper'],s['widths'])],[-float(l/w) for l,w in zip(s['lower'],s['widths'])]]
 # E@P is unchanged;491/525 identity inherited by frozen source/input hashes.
 E=sparse.csr_matrix((1411,n))
 return dict(matrix=sparse.vstack(parts,format='csr'),rhs=rhs,cost=cost,bounds=np.c_[np.r_[np.full(n,-np.inf),last[0]],np.r_[np.full(n,np.inf),last[1]]],equality=sparse.csr_matrix((int(d['eq']['optimizer_keep'].sum()),n+1)),equality_rhs=np.zeros(int(d['eq']['optimizer_keep'].sum())),ceiling=ceiling)

