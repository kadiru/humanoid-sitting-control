"""One bounded partial-residual/native-auxiliary whole-path algorithm.
Every reconstructed original direction is certified before finite encoding.
"""
from low_seat_lifted_execution_492 import *
import subprocess,sys
from pathlib import Path
N0=10540;KIN=np.r_[np.arange(14),46];N=78790
SPEC=dict(SPEC)
SPEC.update(scope=__doc__,coordinate_system='Audited504 mixed15 kinematic-residual and35 nativeauxiliary coordinates perstate, with inherited502 normal-gap incidence.',
 method='highs-ipm',resource_address_space_GiB=40,per_lp_worker_wall_cap_s=120,
 solver_failure_policy='Terminate entire run at first absent/nonoptimal/uncertified LP witness or resource failure. No timeout-driven trust retry or alternative method.',
 rationale='504/505 preserves native commands/torques/COM as direct coordinates and only normalizes the15kinematic rows. All consistency and original rows remain literal. Saved primary coefficient count6099134 is below50223691961/49832076557 but above original4902613498. This justifies a separately bounded complete attempt, not a demonstrated speedup, timeout-cause claim, old-run retry or relaxation.',
 source_504='Exact initial C/h inherits SEALED504 S/a/R and505 subblock certificate without inverse/proof repeats. New changed C requires fresh15x15 inverse and its directed nonsingularity certificate; only changed h updates the affine shift.',
 reconstruction='Original t=S*z+a is formed without edits; transformed and original LP witnesses must pass1e-7, status0 only. Reconstruct complete original direction BEFORE dyadic scaling. Original exact positive interval certificate before490decoder. No15-root or direct assignment of new-coordinate gaps.',
 budget_rationale='ONE distinct24major/48LP/120map process, same45s LP/native1e-9/linear1e-7,120s child wallcap and40GiB addressspace.120maps preservesloadrounding proof. Terminal failure policy disallows numerical-method retries. No old budget extended.',
 initial_coordinate_solves=0,initial_inverse_certificates_repeated=0,physical_admission=False)

def build_basis(C,h,R,T):
    count=len(R);roots=(np.arange(count)[:,None]*50+np.arange(15)).ravel();kinrows=(np.arange(count)[:,None]*50+KIN).ravel();aux=(np.arange(count)[:,None]*50+np.arange(15,50)).ravel()
    RB=sparse.block_diag(list(R),format='csr');E=sparse.csr_matrix((np.ones(len(roots)),(roots,np.arange(len(roots)))),shape=(count*50,len(roots)))
    native=sparse.csr_matrix((np.ones(len(aux)),(aux,aux)),shape=(count*50,count*50));T0=T[:N0,:N0]
    S=sparse.bmat([[T0,sparse.csr_matrix((N0,count*50))],[E@((-RB@C[kinrows,:N0])@T0),E@RB@E.T+native]],format='csr')
    a=np.r_[np.zeros(N0),E@(-RB@h[kinrows])];return S,a

def directed_new_inverse(K,R):
    lo=np.zeros_like(R);hi=np.zeros_like(R)
    for j in range(15):
        product=K[:,:,j,None]*R[:,None,j,:]
        if not np.all(np.isfinite(product)):raise ValueError('Nonfinite new kinematic inverse product')
        lo=np.nextafter(lo+np.nextafter(product,-np.inf),-np.inf);hi=np.nextafter(hi+np.nextafter(product,np.inf),np.inf)
    lo=np.nextafter(lo-np.eye(15),-np.inf);hi=np.nextafter(hi-np.eye(15),np.inf);rows=np.zeros((len(R),15));b=np.maximum(abs(lo),abs(hi))
    for j in range(15):rows=np.nextafter(rows+b[:,:,j],np.inf)
    return lo,hi,rows.max(axis=1)

class Coordinates:
    def __init__(self):
        P=D/'mixed_coordinate_preparation_504';self.C=sparse.load_npz(D/'lifted_feasibility_preparation_490/consistency_chain.npz').tocsr()
        self.S=sparse.load_npz(P/'coordinate_transform.npz').tocsr();z=dict(np.load(P/'coordinates.npz'));self.a=z['shift'];self.R=z['R'];self.h=np.load(D/'lifted_feasibility_preparation_490/reference.npz')['h']
        self.T=sparse.load_npz(D/'sparse_gap_residual_preparation_502/gap_incidence.npz').tocsr();self.origin='sealed504';self.solves=0;self.certificates=0
    def get(self,b,folder):
        C=b['C'];same=C.shape==self.C.shape and np.array_equal(C.indptr,self.C.indptr) and np.array_equal(C.indices,self.C.indices) and np.array_equal(C.data,self.C.data)
        if not same:
            co=C[:,N0:].tocoo();assert np.all(co.row//50==co.col//50);L=np.zeros((1365,50,50));L[co.row//50,co.row%50,co.col%50]=co.data
            assert np.all(L[:,KIN,15:]==0);K=L[:,KIN,:15];R=np.linalg.solve(K,np.broadcast_to(np.eye(15),K.shape));self.solves+=1365
            lower,upper,bound=directed_new_inverse(K,R);self.certificates+=1365
            np.savez_compressed(folder/'new_inverse_certificate.npz',K=K,R=R,lower=lower,upper=upper,bound=bound)
            if not np.all(bound<1):raise ValueError('New15x15 inverse failed nonsingularity certificate')
            S,a=build_basis(C,b['h'],R,self.T);self.C,self.S,self.a,self.h,self.R=C,S,a,b['h'].copy(),R;self.origin=str(folder)
        elif not np.array_equal(b['h'],self.h):
            roots=(np.arange(1365)[:,None]*50+np.arange(15)).ravel();kinrows=(np.arange(1365)[:,None]*50+KIN).ravel();a=np.zeros(N);a[N0+roots]=-sparse.block_diag(list(self.R),format='csr')@b['h'][kinrows];self.a=a;self.h=b['h'].copy()
        sparse.save_npz(folder/'coordinate_transform.npz',self.S);np.savez_compressed(folder/'coordinate_shift.npz',a=self.a)
        write(folder/'coordinate_scope.json',dict(inverse_origin=self.origin,same_C_as_previous=same,local_block_solves_cumulative=self.solves,new_inverse_certificates_cumulative=self.certificates,initial_proof_inherited=True,literal_defects_retained=True,local_inverse_dimension=15,native_auxiliary_coordinates=47775))
        return self.S,self.a

def violation(p,w):
    return max(0.,float((p['matrix']@w-p['rhs']).max()),float(abs(p['equality']@w-p['equality_rhs']).max()),float((p['bounds'][:,0]-w).max()),float((w-p['bounds'][:,1]).max()))

def solve_mixed(folder,d,b,s,S,a,ceiling=None):
    folder.mkdir(exist_ok=False);p=assemble(d,b,s,ceiling)
    for name in ['matrix','equality']:sparse.save_npz(folder/(name+'.npz'),p[name])
    np.savez_compressed(folder/'arrays.npz',**{k:p[k] for k in ['rhs','cost','bounds','equality_rhs']});write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
    T=folder/'transformed';T.mkdir();V=sparse.block_diag([S,sparse.csr_matrix([[1.]])],format='csr');z=dict(p);z['matrix']=(p['matrix']@V).tocsr();z['rhs']=p['rhs']-p['matrix'][:,:-1]@a
    for name in ['matrix','equality']:sparse.save_npz(T/(name+'.npz'),z[name])
    np.savez_compressed(T/'arrays.npz',**{k:z[k] for k in ['rhs','cost','bounds','equality_rhs']})
    start=time.monotonic();timed=False
    with (T/'worker.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_mixed_lp_506.py')),str(T)],stdout=log,stderr=subprocess.STDOUT)
        try:rc=proc.wait(timeout=120)
        except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
    called=(T/'scipy_started.json').exists();meta=read(T/'result.json') if (T/'result.json').exists() else dict(status=-120 if timed else -30,message='LP child wallcap reached' if timed else 'LP child exited before result',candidate=False,success=False,iterations=0,crossover_iterations=0)
    meta.update(actual_lp_call=called,worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='highs-ipm',transformed_nnz=z['matrix'].nnz);meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
    if meta['candidate'] and rc==0 and not timed:
        q=np.load(T/'witness.npy');w=np.r_[S@q[:-1]+a,q[-1]];np.save(folder/'witness.npy',w);v=violation(p,w);tv=violation(z,q)
        meta.update(linear_violation=v,transformed_linear_violation=tv,linear_pass=bool(np.all(np.isfinite(w)) and v<=1e-7 and tv<=1e-7),objective=float(p['cost']@w),reconstruction_unmodified=True)
    meta['transformed_candidate']=meta['candidate'];meta['candidate']=w is not None;meta['terminal_failure']=not (rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False))
    write(folder/'solver.json',meta);return w,meta
