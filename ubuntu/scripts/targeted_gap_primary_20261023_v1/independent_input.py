"""Independent new-state C/interval/full input proof; imports no builder or solver."""
from common import *
from fractions import Fraction as F
import math
import numpy as np
from scipy import sparse
OUT=np.r_[np.arange(14),np.arange(15,51)]
fq=lambda x:F(float(x))
rat=lambda x:F(*x)
def same(a,b):
 assert a.shape==b.shape and a.dtype==b.dtype and a.tobytes()==b.tobytes()
def csr(a,b):
 a=a.tocsr();b=b.tocsr();assert a.shape==b.shape
 same(a.indptr,b.indptr);same(a.indices,b.indices);same(a.data,b.data)

def prove(origin):
 check(origin)
 z=dict(np.load(BASE/'reference.npz'));gl=read(D/'gap_coordinate_preparation_v3_466/layout.json')
 d=dict(z=z,S=sparse.load_npz(BASE/'feature_basis.npz').tocsr(),
   N=sparse.load_npz(BASE/'normal_drop_chain.npz').tocsr(),normal=np.asarray(gl['normal_states']),
   dom=read(D/'exact_gap_chart_preparation_468/layout.json'),
   row=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz')),
   actual=dict(np.load(PRIOR/'actual.npz')),A=sparse.load_npz(STATE/'A.npz').tocsr(),
   mu=read(PRIOR/'decision.json')['summary']['phi'])
 saved=dict(np.load(O/'state.npz'));f=saved['features'];y=saved['actualY'];J=saved['J']
 same(f,np.load(PRIOR/'mapped.npz')['features']);same(y,d['actual']['Y']);same(J,np.load(PRIOR/'accepted_J.npz')['J'])
 im=read(O/'intervals.json');assert im['trust']==1.0 and im['drops']==read(PRIOR/'map.json')['drop_state']
 H,l,u,width,cl,cu,owners=interval(d,f,[rat(x) for x in im['drops']],1.0)
 for key,value in [('lower',l),('upper',u),('widths',width),('closed_lower',cl),('closed_upper',cu)]:
  assert [rat(x) for x in im[key]]==value
 assert owners==im['owners']
 csr(H,sparse.load_npz(O/'H.npz'));csr(H,sparse.load_npz(STATE/'H.npz'))
 check(origin)
 M,rhs,cost,bounds,A,C,r,h,guard=matrix(d,f,y,J,H,l,u,width,cl,cu,None)
 arrays=dict(np.load(O/'full_arrays.npz'));full=sparse.load_npz(O/'full_matrix.npz')
 csr(M,full);csr(A,sparse.load_npz(O/'A.npz'));csr(C,sparse.load_npz(O/'C.npz'))
 for key,value in [('rhs',rhs),('cost',cost),('bounds',bounds)]:same(value,arrays[key])
 for key,value in [('r',r),('h',h),('guard',guard)]:same(value,saved[key])
 same(saved['e'],d['actual']['auxiliary_equalities'])
 E=sparse.load_npz(O/'full_equality.npz')
 assert E.shape==(1366,78791) and E.nnz==0 and np.all(arrays['equality_rhs']==0)
 assert M.shape==(652529,78791) and H.shape==(92267,78790)
 assert np.count_nonzero(cost)==1 and cost[-1]==1 and bounds[-1,0]==0
 assert np.isfinite(M.data).all() and np.isfinite(rhs).all()
 check(origin)
 result=dict(passed=True,new_state_full_input_proved=True,new_interval_rows=92267,
   new_consistency_rows=68250,original_columns=78791,
   original_kept_signed_rows=331495,original_total_signed_rows=331888,
   old_state_reductions_used=False,unchanged_A_P_H_identity_inherited=True,
   completed_old_coefficient_proofs_replayed=False,new_model_calls=0,
   full_shape=list(M.shape),full_nnz=int(M.nnz),zero_face_added=False,
   file_sha256={p.name:sha(p) for p in O.iterdir() if p.suffix in ['.npz','.json'] and p.name not in ['admission.json']})
 write(O/'input_proof.json',result)
 return result
def interval(d,f,drops,trust):
    z=d['z'];S=d['S'];lo=z['lower'];hi=z['upper'];ids=z['varying'];controls=z['control_feature_rows'];v=z['feature_scale'].ravel();flat=f.ravel();ng=set((d['normal']*75+71).tolist());U=F(1,2**59)
    low=[];high=[];cl=[];cu=[];owners=[]
    for j,row in enumerate(list(controls)+list(ids)):
        if j<len(lo):l=float(lo[j]);u=float(hi[j]);owner=['original_control',j]
        else:l=float(flat[row]-trust*v[row]);u=float(flat[row]+trust*v[row]);owner=['feature_trust',int(row)]
        c=fq(flat[row]);cl.append(fq(l)-c);cu.append(fq(u)-c);owners.append(owner)
        if j>=len(lo) and int(row) in ng:low.append(fq(l)-c+207*U);high.append(fq(u)-c-207*U)
        else:low.append(fq(math.nextafter(l,math.inf))-c);high.append(fq(math.nextafter(u,-math.inf))-c)
    for i,(l,u,c) in enumerate(zip(map(rat,d['dom']['lower']),map(rat,d['dom']['upper']),drops)):
        e=(205 if i==d['dom']['pivot'] else 1)*U;cl.append(l-c);cu.append(u-c);low.append(l-c+e);high.append(u-c-e);owners.append(['drop',i])
    width=[u-l for l,u in zip(low,high)];assert all(x>0 for x in width)
    return sparse.vstack([S[controls],S[ids],d['N']],format='csr'),low,high,width,cl,cu,owners

def matrix(d,f,y,J,H,l,u,width,cl,cu,ceiling):
    S=d['S'];row=d['row'];keep=row['keep'];soft=row['soft']&keep;hard=keep&~soft;active=d['z']['active']
    a=d['actual'];r=a['auxiliary_rows'];A=d['A'];h=(a['consistency'][active][:,OUT]/d['z']['sigma'][active][:,OUT]).ravel();ss={'phi':d['mu']}
    blocks=np.zeros((len(active),50,75));blocks[:,:,:24]=-J[active][:,OUT,:]
    for j,k in enumerate(OUT):blocks[:,j,24+k]=1.
    ids=(active[:,None]*75+np.arange(75)).ravel();C=(sparse.block_diag(list(blocks),format='csr')@S[ids]).multiply((1/d['z']['sigma'][active][:,OUT].ravel())[:,None]).tocsr()
    variable=np.diff(A.indptr)>0;guard=np.zeros(len(r));guard[soft&variable]=2.**-20;hp=hard&variable;guard[hp]=np.where(r[hp]>0,np.minimum(2.**-20,.5*r[hp]),2.**-20)
    n=S.shape[1];m=C.shape[0];HN=H.multiply((1/np.array([float(w) for w in width]))[:,None]).tocsr();closed=sparse.hstack([HN,sparse.csr_matrix((H.shape[0],1))]);crhs=np.r_[[float(v/w) for v,w in zip(cu,width)],[-float(v/w) for v,w in zip(cl,width)]]
    if ceiling is None:
        top=sparse.hstack([-A[keep],sparse.csr_matrix(-soft[keep,None].astype(float))]);last=sparse.csr_matrix(-np.ones((m,1)));rhs=np.r_[r[keep]-guard[keep],-h,h,crhs];cost=np.r_[np.zeros(n),1.];boundmax=max(1.,2*ss['phi'])
    else:
        top=sparse.hstack([-A[keep],sparse.csr_matrix((keep.sum(),1))]);last=sparse.csr_matrix((m,1));rhs=np.r_[r[keep]-guard[keep]+soft[keep]*ceiling,-h+ceiling,h+ceiling,crhs];cost=np.r_[np.zeros(n),-1.];boundmax=.5
    closure=sparse.vstack([sparse.hstack([C,last]),sparse.hstack([-C,last])]);parts=[top,closure,closed,-closed]
    if ceiling is not None:
        one=sparse.csr_matrix(np.ones((H.shape[0],1)));parts += [sparse.hstack([HN,one]),sparse.hstack([-HN,one])];rhs=np.r_[rhs,[float(v/w) for v,w in zip(u,width)],[-float(v/w) for v,w in zip(l,width)]]
    bounds=np.c_[np.r_[np.full(n,-np.inf),0.],np.r_[np.full(n,np.inf),boundmax]]
    return sparse.vstack(parts,format='csr'),rhs,cost,bounds,A,C,r,h,guard

