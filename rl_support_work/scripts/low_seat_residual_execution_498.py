"""One bounded residual-coordinate interior-point feasibility algorithm.
Original-direction certification precedes any finite encoding. No root.
"""
from low_seat_lifted_execution_492 import *
from low_seat_residual_coordinates_496 import transform,N0,BLOCK
import subprocess,sys
from pathlib import Path
SPEC=dict(SPEC)
SPEC.update(scope=__doc__,coordinate_system='Audited496 affine local residual coordinates; all literal closure defects retained.',
 method='highs-ipm',resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 solver_failure_policy='Terminate this whole run immediately on absent/nonoptimal/uncertified LP witness or resource failure. No timeout-driven trust retry or alternate method.',
 rationale='Joint normalized residual coordinates directly represent local consistency while retaining complete commands/rates/targets. Explicit primal-dual interior point method is one distinct offline numerical formulation, not a claim about492 timeout cause. Extra sparse fill is retained and bounded by worker resources.',
 source_496='Exact initial C/h uses SEALED496 S/a and497 proof without repeating the inverse/certificate fixture. Changed C requires a newly certified local substitution; changed h alone only changes affine shift.',
 reconstruction='Original t=S*z+a is formed once without edits, and both original/transformed LP witnesses checked<=1e-7. Only status0 witnesses may proceed. Primary is never mapped. Dyadic scaling is applied ONLY to reconstructed original t, followed by unchanged exact interval certification.',
 budget_rationale='ONE new24-major/48-LP/120-map algorithm, unchanged45s native LP and numerical/physical criteria.120s child-process wall cap and40GiB address-space cap prevent unbounded preprocessing/factorization. No old run or budget is extended.',
 initial_coordinate_solves=0,initial_inverse_certificates_repeated=0,physical_admission=False)

def inverse_bound(L,Q):
    lo=np.zeros_like(Q);hi=np.zeros_like(Q)
    for k in range(50):
        x=L[:,:,k,None]*Q[:,None,k,:]
        if not np.all(np.isfinite(x)):raise ValueError('Nonfinite new inverse product')
        lo=np.nextafter(lo+np.nextafter(x,-np.inf),-np.inf);hi=np.nextafter(hi+np.nextafter(x,np.inf),np.inf)
    lo=np.nextafter(lo-np.eye(50),-np.inf);hi=np.nextafter(hi-np.eye(50),np.inf)
    a=np.maximum(abs(lo),abs(hi));rows=np.zeros((len(Q),50))
    for k in range(50):rows=np.nextafter(rows+a[:,:,k],np.inf)
    bound=rows.max(axis=1)
    if not np.all(bound<1):raise ValueError('New inverse nonsingularity certificate failed')
    return lo,hi,bound

class Coordinates:
    def __init__(self):
        P=D/'residual_coordinate_preparation_496';self.C=sparse.load_npz(D/'lifted_feasibility_preparation_490/consistency_chain.npz').tocsr()
        self.S=sparse.load_npz(P/'coordinate_transform.npz').tocsr();a=np.load(P/'shift.npz');self.a=a['a'];self.h=a['h']
        z=np.load(P/'local_blocks.npz');self.Q=z['Q'];self.L=z['L'];self.origin='sealed496';self.solves=0;self.certificates=0
    def get(self,b,folder):
        C=b['C'];same=C.shape==self.C.shape and np.array_equal(C.indptr,self.C.indptr) and np.array_equal(C.indices,self.C.indices) and np.array_equal(C.data,self.C.data)
        if not same:
            co=C[:,N0:].tocoo();assert np.all(co.row//50==co.col//50)
            L=np.zeros((1365,50,50));L[co.row//50,co.row%50,co.col%50]=co.data
            S,a,Q,_,_=transform(C,L,b['h']);lo,hi,bound=inverse_bound(L,Q)
            np.savez_compressed(folder/'new_inverse_certificate.npz',L=L,Q=Q,lower=lo,upper=hi,bound=bound)
            self.C,self.S,self.a,self.h,self.Q,self.L=C,S,a,b['h'].copy(),Q,L;self.origin=str(folder);self.solves+=1365;self.certificates+=1365
        elif not np.array_equal(b['h'],self.h):
            self.a=np.r_[np.zeros(N0),-sparse.block_diag(list(self.Q),format='csr')@b['h']];self.h=b['h'].copy()
        sparse.save_npz(folder/'coordinate_transform.npz',self.S);np.savez_compressed(folder/'coordinate_shift.npz',a=self.a)
        write(folder/'coordinate_scope.json',dict(inverse_origin=self.origin,same_C_as_previous=same,local_block_solves_cumulative=self.solves,new_inverse_certificates_cumulative=self.certificates,initial_proof_inherited=True,literal_defects_retained=True))
        return self.S,self.a

def violation(p,w):
    return max(0.,float((p['matrix']@w-p['rhs']).max()),float(abs(p['equality']@w-p['equality_rhs']).max()),float((p['bounds'][:,0]-w).max()),float((w-p['bounds'][:,1]).max()))

def solve_residual(folder,d,b,s,S,a,ceiling=None):
    folder.mkdir(exist_ok=False);p=assemble(d,b,s,ceiling)
    for name in ['matrix','equality']:sparse.save_npz(folder/(name+'.npz'),p[name])
    np.savez_compressed(folder/'arrays.npz',**{k:p[k] for k in ['rhs','cost','bounds','equality_rhs']});write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
    T=folder/'transformed';T.mkdir();V=sparse.block_diag([S,sparse.csr_matrix([[1.]])],format='csr')
    z=dict(p);z['matrix']=(p['matrix']@V).tocsr();z['rhs']=p['rhs']-p['matrix'][:,:-1]@a
    for name in ['matrix','equality']:sparse.save_npz(T/(name+'.npz'),z[name])
    np.savez_compressed(T/'arrays.npz',**{k:z[k] for k in ['rhs','cost','bounds','equality_rhs']})
    start=time.monotonic();timed=False
    with (T/'worker.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_residual_lp_498.py')),str(T)],stdout=log,stderr=subprocess.STDOUT)
        try:rc=proc.wait(timeout=120)
        except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
    called=(T/'scipy_started.json').exists()
    meta=read(T/'result.json') if (T/'result.json').exists() else dict(status=-120 if timed else -30,message='LP child wall cap reached' if timed else 'LP child exited before returning a result',candidate=False,success=False,iterations=0,crossover_iterations=0)
    meta.update(actual_lp_call=called,worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='highs-ipm',transformed_nnz=z['matrix'].nnz)
    meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
    if meta['candidate'] and rc==0 and not timed:
        q=np.load(T/'witness.npy');w=np.r_[S@q[:-1]+a,q[-1]];np.save(folder/'witness.npy',w)
        v=violation(p,w);tv=violation(z,q);meta.update(linear_violation=v,transformed_linear_violation=tv,linear_pass=bool(np.all(np.isfinite(w)) and v<=1e-7 and tv<=1e-7),objective=float(p['cost']@w),reconstruction_unmodified=True)
    meta['transformed_candidate']=meta['candidate'];meta['candidate']=w is not None
    meta['terminal_failure']=not (rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False))
    write(folder/'solver.json',meta);return w,meta
