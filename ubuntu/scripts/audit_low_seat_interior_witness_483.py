"""Independent saved-array audit of482 interval/witness assembly, not a solver."""
from pathlib import Path
from fractions import Fraction as F
import json,datetime,hashlib,math
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'interior_witness_preparation_482';O=D/'interior_witness_audit_483'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def rat(p):return F(int(p[0]),int(p[1]))
def binary(x):return F.from_float(float(x))
def err(a,b):
    c=(a-b).tocsr();c.eliminate_zeros();return float(abs(c.data).max()) if c.nnz else 0.

def main():
    assert read(D/'interior_preparation_exit_482.json')['actual_exit']==0
    for n,h in read(P/'SHA256.json').items():assert sha(P/n)==h,n
    O.mkdir(exist_ok=False)
    S=D/'free_gap_path_472/linear_00';a=dict(np.load(S/'input.npz'));A=sparse.load_npz(S/'A.npz').tocsr();C=sparse.load_npz(S/'C.npz').tocsr()
    cm=dict(np.load(D/'clock_row_preparation_446/coordinate_map.npz'));rows=dict(np.load(D/'clock_row_preparation_v2_446/row_arrays.npz'))
    B=sparse.load_npz(D/'gap_coordinate_preparation_v3_466/coordinate_basis.npz').tocsr();layout=read(D/'gap_coordinate_preparation_v3_466/layout.json')
    free=cm['free'];ref=cm['reference'].ravel()[free];lo=np.maximum(cm['lower'].ravel()[free],ref-.02);hi=np.minimum(cm['upper'].ravel()[free],ref+.02)
    ids=[int(x) for x in free if int(x)%10];pos={int(v):i for i,v in enumerate(free)};Z=B[ids]
    rr=[];cc=[];vv=[]
    for j,c in enumerate(layout['columns']):
        if c['kind']=='normal_drop':rr.extend([c['edge'],c['pivot']]);cc.extend([j,j]);vv.extend([1.,-1.])
    N=sparse.csr_matrix((vv,(rr,cc)),shape=(206,B.shape[1]));H=sparse.vstack([C,Z,N],format='csr')
    dl=read(D/'exact_gap_chart_preparation_468/layout.json');gl=list(map(rat,dl['lower']));gu=list(map(rat,dl['upper']));drops=list(map(rat,read(S/'metadata.json')['drops']));pivot=dl['pivot'];U=F(1,2**59);rounding=[U]*206;rounding[pivot]=205*U
    total=rat(dl['total']);assert sum(l+e for l,e in zip(gl,rounding))<total<sum(u-e for u,e in zip(gu,rounding))
    checks={}
    for mode in ['predictor','correction']:
        Q=P/mode;meta=read(Q/'intervals.json');saved=dict(np.load(Q/'arrays.npz'));radius=.02
        if mode=='predictor':fl=np.maximum(lo-a['z'],-radius);fu=np.minimum(hi-a['z'],radius)
        else:
            fl=np.maximum.reduce([lo-a['z'],a['origin']-radius-a['z'],np.full(len(lo),-.25*radius)])
            fu=np.minimum.reduce([hi-a['z'],a['origin']+radius-a['z'],np.full(len(lo),.25*radius)])
        tl=np.maximum(lo,a['origin']-radius);tu=np.minimum(hi,a['origin']+radius)
        if mode=='correction':tl=np.maximum(tl,a['z']-.25*radius);tu=np.minimum(tu,a['z']+.25*radius)
        lower=list(map(binary,fl));upper=list(map(binary,fu));closed_l=lower.copy();closed_u=upper.copy();owners=[['full_linear_control',i] for i in range(len(fl))]
        for v in ids:
            b,s=divmod(v,10);i=pos[v];c=binary(a['target'][b,s]);l=tl[i];u=tu[i]
            lower.append(binary(math.nextafter(l,math.inf))-c);upper.append(binary(math.nextafter(u,-math.inf))-c)
            closed_l.append(binary(l)-c);closed_u.append(binary(u)-c);owners.append(['target',b,s])
        lower += [l-c+e for l,c,e in zip(gl,drops,rounding)];upper += [u-c-e for u,c,e in zip(gu,drops,rounding)]
        closed_l += [l-c for l,c in zip(gl,drops)];closed_u += [u-c for u,c in zip(gu,drops)]
        owners += [['drop',i] for i in range(206)];widths=[u-l for l,u in zip(lower,upper)]
        for key,expected in [('lower',lower),('upper',upper),('widths',widths),('closed_lower',closed_l),('closed_upper',closed_u),('drop_encoding_errors',rounding)]:assert list(map(rat,meta[key]))==expected,key
        assert owners==meta['owners'] and all(w>0 for w in widths)
        assert err(H,sparse.load_npz(Q/'interval_matrix.npz'))==0
        keep=rows['keep'];soft=rows['soft']&keep;n=B.shape[1];top=sparse.hstack([-A[keep],sparse.csr_matrix((keep.sum(),1))],format='csr')
        closed=sparse.hstack([H,sparse.csr_matrix((H.shape[0],1))],format='csr');HN=H.multiply((1/np.array([float(w) for w in widths]))[:,None]).tocsr();one=sparse.csr_matrix(np.ones((H.shape[0],1)))
        mat=sparse.vstack([top,closed,-closed,sparse.hstack([HN,one]),sparse.hstack([-HN,one])],format='csr')
        rhs=np.r_[a['r'][keep]-a['guard'][keep]+soft[keep]*meta['soft_ceiling'],list(map(float,closed_u)),[-float(x) for x in closed_l],[float(u/w) for u,w in zip(upper,widths)],[-float(l/w) for l,w in zip(lower,widths)]]
        assert np.array_equal(rhs,saved['rhs']);assert np.array_equal(saved['cost'],np.r_[np.zeros(n),-1.]);assert np.array_equal(saved['bounds'],np.c_[np.r_[np.full(n,-np.inf),0.],np.r_[np.full(n,np.inf),.5]])
        matrix_error=err(mat,sparse.load_npz(Q/'matrix.npz'));assert matrix_error==0
        eq=sparse.load_npz(Q/'equality.npz');assert eq.shape==(1366,n+1) and eq.nnz==0 and np.all(saved['equality_rhs']==0)
        # Independently certify the zero fixture from exact rational intervals.
        eta=min([F(1,2)]+[min(-l,u)/w for l,u,w in zip(lower,upper,widths)])
        expected=read(P/'preparation.json')['checks'][mode]['zero_certificate'];assert rat(expected['certified_relative_clearance'])==eta and expected['passed']==(eta>0)
        checks[mode]=dict(shape=mat.shape,matrix_error=matrix_error,rhs_error=0,interval_error=0,zero_relative_clearance=float(eta))
    cases=read(P/'certificate_cases.json')
    for case in cases:
        v=case['tangent'];e=F(1,2) if isinstance(v,list) else min(binary(v),1-binary(v))
        assert rat(case['result']['certified_relative_clearance'])==e and case['result']['passed']==(e>0)
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,checks=checks,certificate_cases=len(cases),drop_interior_arithmetic_nonempty=True,full_joint_interior_feasibility_unknown=True,dependent_f_actual_feasibility_not_implied=True,exact_certificate_never_edits_witness=True,all_original_and_inward_criteria_preserved=True,model_calls=0,root_calls=0,lp_calls=0,new_complete_path=False,physical_admission=False)
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
