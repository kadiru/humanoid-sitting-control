"""Independent scalar/stdlib audit of the actual-row acceptance rule."""
import math
from fractions import Fraction

def evaluate(mu, phi, predicted, ratio, auxiliary_pass, before, after, hard,
             targets=(214643,215327)):
    assert len(before)==len(after)==len(hard)
    ids=[i for i,selected in enumerate(hard) if selected]
    assert ids and len(targets)==2 and all(i in ids for i in targets)
    finite=all(math.isfinite(float(v)) for v in before) and all(math.isfinite(float(v)) for v in after)
    finite=finite and all(math.isfinite(v) for v in (mu,phi,predicted)) and (ratio is None or math.isfinite(ratio))
    progress=bool(finite and mu>0 and Fraction(phi)<=Fraction(mu)*Fraction(99,100))
    bad=[i for i in ids if float(before[i])>=0 and float(after[i])<0]
    old=max([0.]+[-float(before[i]) for i in ids]);new=max([0.]+[-float(after[i]) for i in ids])
    tg=all(float(after[i])>=0 for i in targets)
    valid=bool(auxiliary_pass and predicted>0 and ratio is not None and ratio>=.1)
    return dict(passed=bool(finite and progress and tg and len(bad)==0 and new<=old and valid),
        finite=bool(finite),one_percent_actual_progress=progress,target_rows=list(targets),
        target_before=[float(before[i]) for i in targets],target_after=[float(after[i]) for i in targets],
        targets_nonnegative=bool(tg),newly_negative_hard_rows=bad,
        worst_hard_violation_before=old,worst_hard_violation_after=new,
        worst_hard_not_increased=bool(new<=old),existing_filter_pass=valid,physical_admission=False)
