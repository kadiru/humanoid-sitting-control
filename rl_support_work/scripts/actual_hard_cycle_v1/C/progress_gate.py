"""Research acceptance only; none of these tests imply mechanical admission."""
from fractions import Fraction

def evaluate(mu, phi, predicted, ratio, auxiliary_pass, before, after, hard,
             targets=(214643,215327)):
    import numpy as np
    before=np.asarray(before);after=np.asarray(after);hard=np.asarray(hard)
    assert before.shape==after.shape==hard.shape and hard.dtype==bool and hard.any()
    assert len(targets)==2 and all(hard[i] for i in targets)
    finite=bool(np.isfinite(before).all() and np.isfinite(after).all()
        and np.isfinite([mu,phi,predicted]).all() and (ratio is None or np.isfinite(ratio)))
    progress=bool(finite and mu>0 and 100*(Fraction(mu)-Fraction(phi))>=Fraction(mu))
    newly_bad=np.flatnonzero(hard & (before>=0) & (after<0)).tolist()
    worst_before=max(0.,-float(before[hard].min()));worst_after=max(0.,-float(after[hard].min()))
    target_pass=bool(all(after[i]>=0 for i in targets))
    filter_pass=bool(auxiliary_pass and predicted>0 and ratio is not None and ratio>=.1)
    return dict(passed=bool(finite and progress and target_pass and not newly_bad
        and worst_after<=worst_before and filter_pass),finite=finite,
        one_percent_actual_progress=progress,target_rows=list(targets),
        target_before=[float(before[i]) for i in targets],target_after=[float(after[i]) for i in targets],
        targets_nonnegative=target_pass,newly_negative_hard_rows=newly_bad,
        worst_hard_violation_before=worst_before,worst_hard_violation_after=worst_after,
        worst_hard_not_increased=bool(worst_after<=worst_before),existing_filter_pass=filter_pass,
        physical_admission=False)
