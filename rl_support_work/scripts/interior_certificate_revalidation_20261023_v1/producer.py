"""Proposed named dual-certificate correction; no solve or primal modification."""
from common import *
from fractions import Fraction

def correct_named_multiplier(marginal,row,expected_hex):
 import numpy as np
 assert set(marginal)=={'inequality','equality','lower','upper'}
 assert all(a.dtype==np.float64 and np.isfinite(a).all() for a in marginal.values())
 assert marginal['inequality'][row].item().hex()==expected_hex
 assert marginal['inequality'][row]>0
 corrected={n:a.copy() for n,a in marginal.items()}
 corrected['inequality'][row]=0.0
 return corrected

def numerical_check(M, E, a, x, marginal):
    import numpy as np
    y = -marginal["inequality"]
    nu = -marginal["equality"]
    upper = -marginal["upper"]
    lower = marginal["lower"]
    assert y.shape == (M.shape[0],) and nu.shape == (E.shape[0],)
    assert upper.shape == lower.shape == x.shape == (78791,)
    row = M @ x - a["rhs"]
    equality = E @ x - a["equality_rhs"]
    stationarity = a["cost"] + M.T @ y + E.T @ nu + upper - lower
    lo = np.isfinite(a["bounds"][:, 0])
    hi = np.isfinite(a["bounds"][:, 1])
    primal = max(0., float(row.max(initial=0)),
                 float(abs(equality).max(initial=0)),
                 float((a["bounds"][:, 0]-x).max()),
                 float((x-a["bounds"][:, 1]).max()))
    dual = float(abs(stationarity).max(initial=0))
    gap = abs(float(a["cost"] @ x + a["rhs"] @ y
                    + a["equality_rhs"] @ nu
                    + a["bounds"][hi, 1] @ upper[hi]
                    - a["bounds"][lo, 0] @ lower[lo]))
    finite = all(np.isfinite(z).all() for z in
                 (x, y, nu, upper, lower, row, equality, stationarity,
                  primal, dual, gap))
    signs = bool((y >= 0).all() and (upper >= 0).all() and (lower >= 0).all()
                 and (upper[~hi] == 0).all() and (lower[~lo] == 0).all())
    metrics = dict(primal=primal, dual=dual, absolute_gap=gap,
                   finite=bool(finite), dual_signs=signs,
                   passed=bool(finite and signs and max(primal, dual, gap) <= 1e-9))
    # Nonfinite metrics terminate rather than being serialized as JSON NaN.
    assert finite, "Nonfinite full-original return"
    np.savez(O / "residuals.npz", row=row, equality=equality,
             stationarity=stationarity)
    write(O / "numerical_checks.json", metrics)
    return metrics

def exact_margin(H, metadata, direction, origin):
    resources = read(HERE / "resources.json")
    f = lambda value: Fraction.from_float(float(value))
    coordinates = list(map(f, direction))
    eta = Fraction(1, 2)
    limiting = None
    bad = []
    max_bits = 0
    with (O / "exact_rows.jsonl").open("x") as output:
        for row in range(H.shape[0]):
            if row % 256 == 0:
                check(origin)
            start, end = H.indptr[row:row+2]
            value = sum((f(a) * coordinates[j]
                         for a, j in zip(H.data[start:end], H.indices[start:end])),
                        Fraction(0))
            low = Fraction(*metadata["lower"][row])
            high = Fraction(*metadata["upper"][row])
            width = Fraction(*metadata["widths"][row])
            assert width > 0
            clearance = min(value-low, high-value)/width
            if clearance < eta:
                eta, limiting = clearance, row
            if clearance <= 0:
                bad.append(row)
            max_bits = max(max_bits, *(max(abs(q.numerator).bit_length(),
                                          q.denominator.bit_length())
                                      for q in (value, clearance)))
            output.write(f"[{row},{value.numerator},{value.denominator}]\n")
    result = dict(passed=eta > 0, exact_eta=[eta.numerator, eta.denominator],
                  nonpositive=bad, limiting_row=limiting, rows=H.shape[0],
                  coefficient_products=H.nnz, observed_max_rational_bits=max_bits,
                  direction_scaled=False, finite_map_performed=False,
                  rows_sha256=sha(O / "exact_rows.jsonl"))
    assert max_bits <= max(resources["clearance_numerator_bits_upper_bound"],
                           resources["clearance_denominator_bits_upper_bound"])
    assert (O / "exact_rows.jsonl").stat().st_size <= resources["exact_trace_total_bytes_upper_bound"]
    write(O / "exact_margin.json", result)
    return result

def main():
 origin=admission('producer');spec=contract()
 for p,h in {**spec['input_sha256'],**spec['runtime_sha256']}.items():assert sha(p)==h,p;check(origin)
 assert sha(REJECTED/'backend_primal.npy')==POINT_SHA
 old=read(REJECTED/'audit.json');attribution=read(POSTMORTEM/'closure.json')
 assert old['passed'] and not old['candidate_accepted'] and old['exact_margin'] is None
 assert attribution['attribution']['records']==1 and attribution['observed_violation']['index']==ROW and attribution['observed_violation']['raw_hex']==RAW_HEX
 proof=read(REJECTED/'input_audit.json');assert proof['passed'] and proof['new_state']
 for n,h in proof['files_sha256'].items():assert sha(REJECTED/n)==h,n;check(origin)
 write(O/'inherited_input_identity.json',dict(passed=True,input_audit_sha256=sha(REJECTED/'input_audit.json'),original_rejection_preserved=True,assembly_replayed=False,ceiling=0.061901382259917585))
 import numpy as np
 import shutil
 from scipy import sparse
 x=np.load(REJECTED/'backend_primal.npy',allow_pickle=False);assert x.shape==(78791,) and x.dtype==np.float64 and np.isfinite(x).all()
 m=dict(np.load(REJECTED/'backend_marginals.npz',allow_pickle=False));corrected=correct_named_multiplier(m,ROW,RAW_HEX)
 np.savez(O/'corrected_marginals.npz',**corrected)
 write(O/'certificate_change.json',dict(row=ROW,source_hex=RAW_HEX,replacement_hex=float(0.).hex(),changed_dual_values=1,changed_primal_values=0,source_primal_sha256=POINT_SHA,policy='One specifically authorized dual entry; no threshold-based/general clipping',no_optimum_claim_from_correction_alone=True))
 M=sparse.load_npz(REJECTED/'matrix.npz').tocsr();E=sparse.load_npz(REJECTED/'equality.npz').tocsr();a=dict(np.load(REJECTED/'arrays.npz',allow_pickle=False))
 assert M.shape==(837063,78791) and M.nnz<=3050574 and E.shape==(1366,78791) and E.nnz==0
 numerical=numerical_check(M,E,a,x,corrected)
 if not numerical['passed']:
  write(O/'candidate.json',dict(accepted=False,reason='Corrected certificate failed unchanged full-original numerical gates',no_retry=True));return
 H=sparse.load_npz(PRIOR/'H.npz').tocsr();assert H.shape==(92267,78790) and H.nnz==125867
 exact=exact_margin(H,read(PRIOR/'intervals.json'),x[:-1],origin)
 accepted=bool(exact['passed'])
 if accepted:shutil.copyfile(REJECTED/'backend_primal.npy',O/'witness.npy')
 write(O/'candidate.json',dict(accepted=accepted,reason='Unchanged point and corrected certificate pass numerical and exact margin gates' if accepted else 'Nonpositive exact margin',no_retry=True,numerical_eta=float(x[-1]),solver_calls=0,physical_admission=False))
 event('producer_completed',origin)
if __name__=='__main__':main()
