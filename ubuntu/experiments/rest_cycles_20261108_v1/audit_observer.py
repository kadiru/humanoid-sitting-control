"""Independent counts/decision audit of the one frozen observer experiment."""
import json
from pathlib import Path
import numpy as np


def main(out):
    r=json.loads((out/'observer_result.json').read_text());checks={}
    checks['whole_start_disjoint']=not (set(r['fit'])&set(r['calibration']) or set(r['fit'])&set(r['test']) or set(r['calibration'])&set(r['test']))
    checks['proper_alignment']=r['alignment'].startswith('obs[t] -> support_bw[t-1]')
    with np.load(out/'observer_predictions.npz',allow_pickle=False) as z:
        for name,t in r['tests'].items():
            a={k.split('__',1)[1]:z[k] for k in z.files if k.startswith(name+'__')}
            lo,hi,y=a['lower'],a['upper'],a['truth'];valid=a['admissible']
            declared_positive=valid & (lo>=.05)
            declared_negative=valid & (hi<.01)
            independently=dict(false_loaded=int((declared_positive&(y<.05)).sum()),
                false_unloaded=int((declared_negative&(y>=.01)).sum()),
                interval_misses=int(((y<lo)|(y>hi))[valid].sum()),
                declared_loaded=int(declared_positive.sum()),declared_unloaded=int(declared_negative.sum()))
            checks[name+'_counts']=all(independently[k]==t[k] for k in independently)
            checks[name+'_no_unknown_declaration']=not (np.any(a['loaded']&~valid) or np.any(a['unloaded']&~valid))
            checks[name+'_finite_ordered_interval']=bool(np.isfinite(lo).all() and np.isfinite(hi).all() and (lo<=hi).all())
            with np.load(Path(t['path']),allow_pickle=False) as original:
                checks[name+'_labels']=bool(np.array_equal(y,original['support_bw'][:-1].reshape(-1)))
    t=r['tests']['demo_rear_back_yaw_right']
    expected=t['loaded_recall']>=.9 and t['unloaded_recall']>=.9 and all(
        a['false_loaded']==a['false_unloaded']==a['interval_misses']==0 for a in r['tests'].values())
    checks['honest_decision']=r['saved_data_screen_passed']==expected and not r['deployment_admitted']
    audit=dict(passed=all(checks.values()),checks=checks,
               meaning='Correct accounting of saved observer result; not successful contact estimation or physical-cycle admission.')
    (out/'observer_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    assert audit['passed'],audit


if __name__=='__main__':
    import sys
    main(Path(sys.argv[1]))
