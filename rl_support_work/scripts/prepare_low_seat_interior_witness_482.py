"""Prepare NEW all-row interior-witness algebra; zero LP/model/root calls."""
import json,datetime
from fractions import Fraction as F
import numpy as np
from scipy import sparse
from low_seat_interior_witness_482 import *
O=D/'interior_witness_preparation_482'

def main():
    O.mkdir(exist_ok=False);d,a=load_reference();checks={}
    for name in ['free_gap_path_472','free_gap_path_audit_473','joint_ray_path_478','joint_ray_path_audit_479']:
        for n,h in read(D/name/'SHA256.json').items():assert sha(D/name/n)==h,n
    # This ceiling is an ASSEMBLY FIXTURE: optimistic primary epigraph zero,
    # half its modeled reduction. No LP was solved and no witness is implied.
    mu=max(0.,float(-a['r'][d['a']['soft']&d['a']['keep']].min()));ceiling=.5*mu
    for mode in ['predictor','correction']:
        b=dict(a);b['mode']=mode
        if mode=='correction':
            radius=.02*b['trust']
            b['full_step_lower']=np.maximum.reduce([d['lower']-b['z'],b['origin']-radius-b['z'],np.full(len(b['z']),-.25*radius)])
            b['full_step_upper']=np.minimum.reduce([d['upper']-b['z'],b['origin']+radius-b['z'],np.full(len(b['z']),.25*radius)])
        w=assemble_witness(d,b,ceiling);save(w,O/mode)
        zero=certify(w['system'],np.zeros(w['matrix'].shape[1]-1))
        checks[mode]=dict(shape=w['matrix'].shape,finite_intervals=w['system']['H'].shape[0],target_intervals=w['system']['target_rows'],zero_certificate=zero)
    # NEW exact-certificate tests, not old finite-map or derivative fixtures.
    cases=[]
    for label,v,passed in [('interior',.5,True),('exact_lower',0.,False),('exact_upper',1.,False),('tiny_outward',-1e-13,False)]:
        s=dict(H=sparse.csr_matrix([[1.]]),lower=[F(0)],upper=[F(1)],widths=[F(1)],owners=[['synthetic',0]])
        c=certify(s,np.array([v]));assert c['passed']==passed;cases.append(dict(label=label,tangent=v,result=c))
    # A coupled exact dot product, rather than a rounded ndarray product.
    s=dict(H=sparse.csr_matrix([[1.,-1.]]),lower=[F(-1)],upper=[F(1)],widths=[F(2)],owners=[['synthetic_coupled',0]])
    c=certify(s,np.array([.3,.3]));assert c['certified_relative_clearance']==[1,2]
    cases.append(dict(label='exact_coupled_cancellation',tangent=[.3,.3],result=c))
    (O/'certificate_cases.json').write_text(json.dumps(cases,indent=2))
    out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,specification=SPEC,checks=checks,certificate_cases=len(cases),assembly_fixture_soft_ceiling=ceiling,fixture_ceiling_is_not_solved_primary_result=True,all331888_final_rows_retained=True,all11906_full_boxes_and206_drop_boxes_retained=True,all_original_buffers_unchanged=True,exact_certificate_never_edits_witness=True,positive_interior_feasibility_unknown=True,model_calls=0,root_calls=0,lp_calls=0,new_complete_path=False,physical_admission=False)
    (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2))
    print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
