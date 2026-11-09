"""Bounded full-system witness construction using exact row implications.

The selected rows are a proved equivalent system, not a partial master.
Every returned vector is still checked against BOTH complete LPs.
"""
from low_seat_mixed_execution_506 import *
from low_seat_exact_row_quotient_v2_516 import quotient
from fractions import Fraction
import shutil

SPEC=dict(SPEC)
SPEC.update(scope=__doc__,rationale='ONE new complete attempt using516/517 exact positive-ray implications rather than506 simultaneous redundant rows or512 deferred cuts. Primary452522rows/5663317mixednonzeros; witness reference618101/6668262. The coefficient reduction is modest and is not evidence of speedup or a timeout cause. All omitted constraints have exact implication witnesses in BOTH binary64systems, and full numerical checks remain mandatory.',
 row_reduction='Union of original and transformed positive-ray representatives; only exact tautologies and dominated rows omitted. No threshold, row scaling change, cut queue, removed consistency sign or old partial vector. Every68250consistency row BOTHsigns remains explicit in the inherited reference.',
 quotient_inheritance='Compare every full matrix coefficient, RHS, objective, bound and equality against the saved516 inputs. An EXACT matching primary or witness inherits its sealed516 rowsets/ledgers and successful517 real-system proof by hash; never repeat the grouping/proof on that system. The failed synthetic fixture is excluded and remains failed.',
 new_systems='A genuinely different full LP uses the unchanged516 exact-integer grouping utility, then an exact rational implication check before solving. Save all ledgers and proof flags. Any proof failure is terminal. A new actual state or actual witness ceiling cannot reuse a stale subset. No optimistic fixture ceiling is used without an actual verified primary result.',
 numerical_witness='Only optimal status0 may be reconstructed as ORIGINALt=S*z+a. Save full original and transformed residual vectors; both complete LP inequalities/equalities/bounds<=1e-7 are mandatory, regardless of reduced residual. No tolerance multiplication by implication factors or vector edits. A nonoptimal internal vector, if returned, remains raw child evidence only.',
 failure_policy='First absent/nonoptimal/uncertified complete witness or resource failure terminates the entire run. No cut additions, fallback, alternate solver, timeout-driven trust retry or old-budget extension. Nonpositive primary modeled improvement also terminates.',
 bounded_resources='24major/48actualLP/120maps,45s nativehighs-ipm1e-9,40GiBchild/120swall; same2^-20guards/trust1..1/32/ratio>=.1 No correction LP is used: five shared dyadic trials1..1/16 per complete interior witness. Trust doubles<=1 only ratio>.75 and halves only after actual finite/nonlinear rejection. No resource or physical threshold expanded.',
 budget_rationale='At most two proved full-system LPs permajor, no cut rounds. At most5 finite maps permajor retains the120-map load-rounding contract. This independently bounded algorithm starts ONLY sealed472INITIAL, not any prior optimizer state.',
 initial_row_quotient_proofs_repeated=0,initial_coordinate_solves=0,physical_admission=False)

def same_matrix(A,B):
    if A.shape!=B.shape:return False
    A=A.tocsr(copy=True);B=B.tocsr(copy=True);A.sort_indices();B.sort_indices()
    return np.array_equal(A.indptr,B.indptr) and np.array_equal(A.indices,B.indices) and np.array_equal(A.data,B.data)

def same_problem(p,folder):
    arrays=dict(np.load(folder/'arrays.npz'))
    return all(np.array_equal(p[k],arrays[k]) for k in ['rhs','cost','bounds','equality_rhs']) and all(same_matrix(p[k],sparse.load_npz(folder/(k+'.npz'))) for k in ['matrix','equality'])

def prove_new_system(M,b,reps,ids):
    """Exact check on a NEW LP only; no old517 proof or synthetic fixture."""
    selected=set(map(int,ids));verified=np.zeros(len(b),dtype=np.uint8)
    for i,j in enumerate(reps):
        st,en=M.indptr[i:i+2];x={int(k):Fraction(float(v)) for k,v in zip(M.indices[st:en],M.data[st:en]) if v!=0}
        if j<0:assert not x and Fraction(float(b[i]))>=0;verified[i]=1;continue
        assert int(j) in selected
        if i==j:verified[i]=1;continue
        st,en=M.indptr[j:j+2];y={int(k):Fraction(float(v)) for k,v in zip(M.indices[st:en],M.data[st:en]) if v!=0}
        assert x and x.keys()==y.keys();k=min(x);ratio=x[k]/y[k];assert ratio>0
        assert all(x[k]==ratio*y[k] for k in x)
        assert Fraction(float(b[i]))>=ratio*Fraction(float(b[j]));verified[i]=1
    assert np.all(verified);return verified

class RowQuotients:
    def __init__(self):self.inherited_systems=0;self.new_systems=0;self.new_proof_rows=0
    def get(self,folder,p,z,role):
        P=D/'exact_row_quotient_preparation_516'/role
        inherited=same_problem(p,D/'lifted_feasibility_preparation_490'/role) and same_problem(z,D/'mixed_coordinate_preparation_504'/role)
        folder.mkdir(exist_ok=False)
        if inherited:
            assert read(D/'exact_row_quotient_audit_517/independent_audit.json')['all_omitted_rows_exactly_implied_in_both_systems']
            for n in ['row_ids.npy','original_ledger.npz','transformed_ledger.npz']:shutil.copy2(P/n,folder/n)
            ids=np.load(folder/'row_ids.npy');self.inherited_systems+=2
        else:
            ledgers={name:quotient(q['matrix'],q['rhs']) for name,q in [('original',p),('transformed',z)]}
            ids=np.union1d(ledgers['original']['representatives'],ledgers['transformed']['representatives']);np.save(folder/'row_ids.npy',ids)
            for name,q in [('original',p),('transformed',z)]:
                ledger=ledgers[name];np.savez_compressed(folder/(name+'_ledger.npz'),**ledger)
                flags=prove_new_system(q['matrix'],q['rhs'],ledger['representative'],ids)
                np.save(folder/(name+'_new_proof_flags.npy'),flags);self.new_proof_rows+=len(flags)
            self.new_systems+=2
        scope=dict(inherited=inherited,role=role,full_rows=p['matrix'].shape[0],selected_rows=len(ids),omitted_rows=p['matrix'].shape[0]-len(ids),original_full_nnz=p['matrix'].nnz,transformed_full_nnz=z['matrix'].nnz,original_selected_nnz=p['matrix'][ids].nnz,transformed_selected_nnz=z['matrix'][ids].nnz,proof_source=str(D/'exact_row_quotient_audit_517') if inherited else 'new exact rational implication flags',old_proof_repeated=False,full_numerical_checks_mandatory=True,synthetic_fixture_used=False)
        write(folder/'scope.json',scope);return ids,scope

def solve_quotient(folder,d,b,s,S,a,quotients,ceiling=None):
    folder.mkdir(exist_ok=False);p=assemble(d,b,s,ceiling)
    for k in ['matrix','equality']:sparse.save_npz(folder/(k+'.npz'),p[k])
    np.savez_compressed(folder/'arrays.npz',**{k:p[k] for k in ['rhs','cost','bounds','equality_rhs']});write(folder/'stage.json',dict(ceiling=ceiling,primary=ceiling is None))
    T=folder/'transformed';T.mkdir();V=sparse.block_diag([S,sparse.csr_matrix([[1.]])],format='csr');z=dict(p);z['matrix']=(p['matrix']@V).tocsr();z['rhs']=p['rhs']-p['matrix'][:,:-1]@a
    for k in ['matrix','equality']:sparse.save_npz(T/(k+'.npz'),z[k])
    np.savez_compressed(T/'arrays.npz',**{k:z[k] for k in ['rhs','cost','bounds','equality_rhs']})
    ids,scope=quotients.get(folder/'quotient',p,z,'primary' if ceiling is None else 'witness')
    child=T/'reduced';child.mkdir();sparse.save_npz(child/'matrix.npz',z['matrix'][ids]);sparse.save_npz(child/'equality.npz',z['equality'])
    np.savez_compressed(child/'arrays.npz',rhs=z['rhs'][ids],cost=z['cost'],bounds=z['bounds'],equality_rhs=z['equality_rhs'])
    start=time.monotonic();timed=False
    with (child/'worker.log').open('w') as log:
        proc=subprocess.Popen([sys.executable,str(Path(__file__).with_name('solve_low_seat_quotient_lp_518.py')),str(child)],stdout=log,stderr=subprocess.STDOUT)
        try:rc=proc.wait(timeout=120)
        except subprocess.TimeoutExpired:timed=True;proc.kill();rc=proc.wait()
    called=(child/'scipy_started.json').exists();meta=read(child/'result.json') if (child/'result.json').exists() else dict(status=-120 if timed else -30,message='LP child wallcap reached' if timed else 'LP child exited before result',candidate=False,success=False,iterations=0,crossover_iterations=0)
    meta.update(actual_lp_call=called,worker_actual_exit=int(rc),worker_wall_timeout=timed,worker_elapsed_s=time.monotonic()-start,method='highs-ipm',transformed_nnz=z['matrix'].nnz,reduced_nnz=z['matrix'][ids].nnz,full_rows=z['matrix'].shape[0],reduced_rows=len(ids),quotient=scope);meta.setdefault('elapsed_s',meta['worker_elapsed_s']);w=None
    raw_candidate=meta['candidate']
    if raw_candidate and meta['status']==0 and rc==0 and not timed:
        q=np.load(child/'witness.npy');np.save(T/'witness.npy',q);w=np.r_[S@q[:-1]+a,q[-1]];np.save(folder/'witness.npy',w)
        ro=p['matrix']@w-p['rhs'];rt=z['matrix']@q-z['rhs'];np.savez_compressed(folder/'full_residuals.npz',original=ro,transformed=rt)
        v=violation(p,w);tv=violation(z,q)
        meta.update(linear_violation=v,transformed_linear_violation=tv,linear_pass=bool(np.all(np.isfinite(w)) and np.all(np.isfinite(q)) and v<=1e-7 and tv<=1e-7),objective=float(p['cost']@w),reconstruction_unmodified=True)
    meta['transformed_candidate']=raw_candidate;meta['candidate']=w is not None;meta['terminal_failure']=not (rc==0 and not timed and meta['status']==0 and w is not None and meta.get('linear_pass',False))
    write(folder/'solver.json',meta);return w,meta
