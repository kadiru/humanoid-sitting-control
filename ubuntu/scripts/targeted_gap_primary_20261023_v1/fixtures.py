"""Synthetic data only; no study files, optimizer or mechanics calls."""
import numpy as np
from scipy import sparse
from target_rows import build
import independent_targets as it
from numerical_checks import check_point,added_duals_zero
from independent_return import independent_check
J=np.zeros((1410,51,24));J[622,47,0]=2;J[1273,47,0]=3;J[1400,47,2]=4;J[631,47,3]=5
P=sparse.csr_matrix((np.array([1.,1.,.5,2.]),(np.array([622*75,1273*75,1400*75+2,631*75+3]),np.array([1,2,3,4]))),shape=(105750,78790))
actual=np.zeros(331888);actual[214643]=-1.;actual[215327]=-.25;scale=np.ones(331888);scale[214643]=.5;scale[215327]=2
G,r=build(P,J,actual,scale)
expected=sparse.csr_matrix(([4.,-6.,1.,-5.],([0,0,1,1],[1,2,3,4])),shape=(2,78791))
assert (G!=expected).nnz==0
assert np.array_equal(r,np.array([-1.-2.**-20,-.25-2.**-20]))
it.read=lambda p:dict(records=[dict(row=214643,group='mesh_normal_descent',scale=.5,actual_after=-1.,terms=[dict(state=622,coefficient=-1.),dict(state=1273,coefficient=1.)]),dict(row=215327,group='mesh_normal_descent',scale=2.,actual_after=-.25,terms=[dict(state=1400,coefficient=-1.),dict(state=631,coefficient=1.)])])
assert it.prove_targets(P,J,actual,scale,G,r)['passed']
M=sparse.csr_matrix([[-1.,0.],[0.,-1.]]);E=sparse.csr_matrix((0,2));a=dict(rhs=np.array([-1.,-.25]),cost=np.array([0.,1.]),bounds=np.array([[-np.inf,np.inf],[0.,1.]]),equality_rhs=np.array([]));w=np.array([1.,.25]);m=dict(inequality=np.array([0.,-1.]),equality=np.array([]),upper=np.zeros(2),lower=np.zeros(2))
assert check_point(M,E,a,w,m)[0]==independent_check(M,E,a,w,m)[0]
assert check_point(M,E,a,w,m)[0]['passed']
assert added_duals_zero(np.array([0.,-0.]))
assert not added_duals_zero(np.array([0.,1e-30]))
assert not added_duals_zero(np.array([0.,-1e-30]))
T=sparse.vstack([M,sparse.csr_matrix([[-1.,0.],[1.,0.]])],format='csr');aa=dict(a,rhs=np.r_[a['rhs'],[-.5,2.]])
mm=dict(m,inequality=np.r_[m['inequality'],[0.,-0.]])
assert it.prove_augmented(T,aa,M,a,T[-2:],np.array([-.5,2.]))['original_prefix_exact']
assert check_point(T,E,aa,w,mm)[0]['passed']
assert independent_check(T,E,aa,w,mm)[0]['passed']
bad=dict(m,inequality=np.array([1e-30,-1.]));assert not check_point(M,E,a,w,bad)[0]['passed']
print('PASS: targeted tangent signs/scaling/basis, independent target assembly, original/augmented KKT, exact-zero added-dual gate including signed zero/tiny nonzero rejection; synthetic only')
