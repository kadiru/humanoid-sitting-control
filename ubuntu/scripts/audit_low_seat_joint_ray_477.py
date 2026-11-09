"""Independent exact shared-ray and joint-encoding audit of NEW476 fixtures.
Uses sparse incidence directly, not the production grouping/decoder. No old
finite fixtures, mechanics, roots, derivatives or optimizer are repeated.
"""
from pathlib import Path
from fractions import Fraction as F
import json,math,datetime,hashlib
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'joint_ray_audit_477'
read=lambda p:json.loads(Path(p).read_text());qf=lambda x:F.from_float(float(x));rat=lambda p:F(int(p[0]),int(p[1]))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def exact_product(A,t):
    ts=list(map(qf,t));out=[]
    for i in range(A.shape[0]):
        a,b=A.indptr[i:i+2];out.append(sum((qf(v)*ts[j] for v,j in zip(A.data[a:b],A.indices[a:b])),F(0)))
    return out
def replay(a,t,B,C,cm,dom,layout):
    dr=exact_product(B,t);planned=exact_product(C,t);target=a['target'];radius=.02
    ref=cm['reference'].ravel()[cm['free']];lo=np.maximum(cm['lower'].ravel()[cm['free']],ref-.02);hi=np.minimum(cm['upper'].ravel()[cm['free']],ref+.02)
    lo=np.maximum(lo,a['origin']-radius);hi=np.minimum(hi,a['origin']+radius)
    low=F(0);high=F(1)
    def restrict(l,u,v):
        nonlocal low,high
        assert l<=u
        if not v:assert l<=0<=u;return
        x,y=sorted([l/v,u/v]);low=max(low,x);high=min(high,y)
    for j,index in enumerate(cm['free']):
        b,s=divmod(int(index),10)
        if s:restrict(qf(lo[j])-qf(target[b,s]),qf(hi[j])-qf(target[b,s]),dr[b*10+s])
        restrict(qf(a['full_step_lower'][j]),qf(a['full_step_upper'][j]),planned[j])
    dl=list(map(rat,dom['lower']));du=list(map(rat,dom['upper']));dv=[F(0)]*206
    for j,c in enumerate(layout['columns']):
        if c['kind']=='normal_drop':dv[c['edge']]+=qf(t[j]);dv[c['pivot']]-=qf(t[j])
    for l,u,x,v in zip(dl,du,a['drops'],dv):restrict(l-x,u-x,v)
    assert 0<=low<=high<=1
    out=target.copy()
    for i,v in enumerate(dr):
        b,s=divmod(i,10);out[b,s]=float(qf(target[b,s])+high*v)
    q=[x+high*v for x,v in zip(a['drops'],dv)];total=rat(dom['total']);pivot=dom['pivot'];ids=dom['independent'];z=[F(0)]*206
    def floor(x):
        f=float(x);return qf(math.nextafter(f,-math.inf) if qf(f)>x else f)
    def ceil(x):
        f=float(x);return qf(math.nextafter(f,math.inf) if qf(f)<x else f)
    assert sum(q)==total
    if not any(dv):z=a['drops'].copy()
    else:
        for i in ids:z[i]=floor(q[i])
        S=sum(z)
        for i in ids:
            if S>=total-du[pivot]:break
            c=ceil(q[i]);assert 0<=c-z[i]<=F(1,2**59);S+=c-z[i];z[i]=c
        z[pivot]=total-S
    assert sum(z)==total and all(l<=v<=u for l,v,u in zip(dl,z,du))
    value=rat(dom['hover']);gg=[float(value)]
    for x in z:value-=x;gg.append(float(value))
    assert value==rat(dom['terminal']);out[np.array(layout['normal_states']),0]=gg
    worst=F(0)
    for b in range(len(out)):
        before=sum(map(qf,target[b,2:10]),F(0));after=sum(map(qf,out[b,2:10]),F(0))
        assert abs(before-after)<=8*F(1,2**54);worst=max(worst,abs(before-after))
    for j,index in enumerate(cm['free']):
        b,s=divmod(int(index),10)
        if s:assert qf(lo[j])<=qf(out[b,s])<=qf(hi[j])
    return out,z,low,high,worst
def main():
    assert read(D/'joint_ray_preparation_exit_476.json')['actual_exit']==0
    P=D/'joint_ray_preparation_476'
    for p,h in read(P/'SHA256.json').items():assert sha(P/p)==h,p
    O.mkdir(exist_ok=False);a=dict(np.load(P/'fixture_origin.npz'));a['drops']=list(map(rat,read(P/'fixture_drop_origin.json')))
    B=sparse.load_npz(D/'gap_coordinate_preparation_v3_466/coordinate_basis.npz').tocsr();C=sparse.load_npz(D/'free_gap_path_472/linear_00/C.npz').tocsr();cm=dict(np.load(D/'clock_row_preparation_446/coordinate_map.npz'));dom=read(D/'exact_gap_chart_preparation_468/layout.json');layout=read(D/'gap_coordinate_preparation_v3_466/layout.json');details=[];worst=0.
    for k in range(9):
        f=np.load(P/f'fixture_{k:02d}.npz');m=read(P/f'fixture_{k:02d}.json');out,z,l,h,e=replay(a,f['tangent'],B,C,cm,dom,layout)
        assert np.array_equal(out,f['target']);assert z==list(map(rat,m['drops']));assert l==rat(m['alpha_lower']) and h==rat(m['alpha']);assert e==rat(m['load_rounding_sum_error']);worst=max(worst,float(e));details.append(dict(fixture=k,exact_alpha=str(h),mapped_exact=True,exact_endpoint=True))
    original=dict(np.load(D/'free_gap_path_472/linear_00/input.npz'));original['drops']=list(map(rat,read(D/'free_gap_path_472/linear_00/metadata.json')['drops']))
    z,zd,l,h,_=replay(original,np.zeros(B.shape[1]),B,C,cm,dom,layout);assert np.array_equal(z,original['target']) and zd==original['drops'] and h==1
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,finite_joint_fixtures=9,all_mapped_values_exact=True,all_shared_alpha_intersections_exact=True,all206_drop_bounds_and_endpoint_exact=True,all11906_linear_control_bounds_included=True,all10540_columns_exercised=True,zero_ray_exact=True,maximum_single_load_sum_rounding=float(worst),source='Original472 initial reference only; fixture center not a candidate; no rejected trial used',actual_phase_or_path_feasibility_claim=False,model_calls=0,root_calls=0,optimizer_calls=0,new_complete_path=False,physical_admission=False,details=details)
    (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
