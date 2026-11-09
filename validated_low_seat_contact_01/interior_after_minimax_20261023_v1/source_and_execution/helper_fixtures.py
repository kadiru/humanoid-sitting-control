"""Synthetic checks only: no study inputs, optimizer or remote output directory."""
import ast
import json
from pathlib import Path
import sys
import tempfile
import time
from fractions import Fraction as F
import numpy as np
from scipy import sparse
import common
import builder
import producer
import independent_return as auditor


def run():
    checks=[]
    try:
        common.contract()
        raise AssertionError('Unapproved contract accepted')
    except PermissionError:
        checks.append('unapproved_contract_blocked_before_science')
    # One kept auxiliary row, two consistency signs, two original closed rows.
    full=sparse.csr_matrix([[1.,2.,-1.],[0.,3.,-1.],[0.,-3.,-1.],[2.,0.,0.],[-2.,0.,0.]])
    arrays=dict(rhs=np.array([.125,-.25,.25,1.125,1.125]),equality_rhs=np.zeros(1))
    state=dict(r=np.array([.5]),guard=np.array([.125]),h=np.array([.25]))
    rows=dict(keep=np.array([True]),soft=np.array([True]))
    interval=dict(lower=[[-1,1]],upper=[[1,1]],widths=[[2,1]])
    M,a=builder.construct(full,arrays,state,rows,.375,interval,1,1)
    expected=np.array([[1,2,0],[0,3,0],[0,-3,0],[2,0,0],[-2,0,0],[2,0,1],[-2,0,1]],dtype=float)
    assert np.array_equal(M.toarray(),expected)
    assert np.array_equal(a['rhs'],[.75,.125,.625,1.125,1.125,.5,.5])
    assert np.array_equal(a['cost'],[0,0,-1]) and np.array_equal(a['bounds'][-1],[0,.5])
    assert np.isneginf(a['bounds'][:2,0]).all() and np.isposinf(a['bounds'][:2,1]).all()
    checks.append('new_ceiling_replaces_epigraph_and_retains_closed_rows')
    with tempfile.TemporaryDirectory(prefix='interior_source_fixture_') as temporary:
        root=Path(temporary)
        cases=[('ordinary_positive',sparse.csr_matrix([[2.,-1.],[0.,1.]]),np.array([.25,-.125]),[F(-1),F(-1)],[F(1),F(1)],True),
               ('exact_boundary_rejected',sparse.csr_matrix([[1.]]),np.array([0.]),[F(0)],[F(1)],False),
               ('outside_rejected',sparse.csr_matrix([[1.]]),np.array([np.nextafter(0.,-1.)]),[F(0)],[F(1)],False),
               ('subnormal_product_preserved',sparse.csr_matrix([[np.nextafter(0.,1.)]]),np.array([np.nextafter(0.,1.)]),[F(0)],[F(1,2**2147)],True)]
        for name,H,d,lo,hi,want in cases:
            out=root/name;out.mkdir();producer.O=out;auditor.O=out
            pairs=lambda v:[[x.numerator,x.denominator] for x in v]
            im=dict(lower=pairs(lo),upper=pairs(hi),widths=pairs([u-l for u,l in zip(hi,lo)]))
            origin=time.monotonic()
            result=producer.exact_margin(H,im,d,origin)
            checked=auditor.audit_exact(H,im,np.r_[d,.25],origin)
            assert result==checked and result['passed']==want
            checks.append(name)
            if name=='subnormal_product_preserved':
                trace=json.loads((out/'exact_rows.jsonl').read_text());assert F(trace[1],trace[2])==F(1,2**2148)
                trace[1]+=1;(out/'exact_rows.jsonl').write_text(json.dumps(trace)+'\n')
                try:auditor.audit_exact(H,im,np.r_[d,.25],origin)
                except AssertionError:checks.append('independent_audit_rejects_trace_corruption')
                else:raise AssertionError('Corrupt trace accepted')
        n=78791;nr=837063;ne=1366
        M=sparse.csr_matrix((nr,n));E=sparse.csr_matrix((ne,n))
        point=np.zeros(n);point[-1]=.5
        cost=np.zeros(n);cost[-1]=-1
        bounds=np.c_[np.full(n,-np.inf),np.full(n,np.inf)];bounds[-1]=[0,.5]
        marginal=dict(inequality=np.zeros(nr),equality=np.zeros(ne),upper=np.zeros(n),lower=np.zeros(n));marginal['upper'][-1]=-1
        for bad in (False,True):
            out=root/('kkt_invalid' if bad else 'kkt_valid');out.mkdir();producer.O=out;auditor.O=out
            rhs=np.zeros(nr)
            if bad:rhs[0]=-1e-8
            arrays=dict(rhs=rhs,cost=cost,bounds=bounds,equality_rhs=np.zeros(ne))
            prod=producer.numerical_check(M,E,arrays,point,marginal)
            indep=auditor.audit_kkt(M,E,arrays,point,marginal)
            assert prod==indep and prod['passed']==(not bad)
            checks.append('full_shape_KKT_violation_rejected' if bad else 'full_shape_KKT_valid_point')
    return dict(passed=True,checks=checks,count=len(checks),scientific_data_loaded=False,solver_calls=0)

if __name__=='__main__':print(json.dumps(run(),indent=2))
