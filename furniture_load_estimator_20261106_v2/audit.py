"""Independent whole-start held-out accuracy and trigger admission."""
import json
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
    with np.load(HERE/'dataset.npz') as z:d=dict(z)
    with np.load(HERE/'predictions.npz') as z:p=dict(z)
    rows=json.loads((HERE/'inputs.json').read_text());results=[]
    for g,row in enumerate(rows):
        if row['split']!='test':continue
        mask=(d['group']==g)&d['valid'];ix=np.flatnonzero(mask)
        assert len(ix)>0
        e=abs(p['mean'][ix]-d['y'][ix]);last=ix[-250:]
        lo=p['mean'][ix,0]-p['radius'][ix,0]
        false_load=int(np.sum((lo>=.05)&(d['y'][ix,0]<.01)))
        pos=d['y'][ix,0]>.12
        recall=float(np.mean(lo[pos]>=.05)) if pos.any() else None
        checks=dict(mean_error=bool((e.mean(0)<=.02).all()),tail_error=bool((np.quantile(e,.99,axis=0)<=.05).all()),
            final_hold_error=bool((abs(p['mean'][last]-d['y'][last]).mean(0)<=.02).all()),no_false_load=false_load==0,
            loaded_recall=recall is None or recall>=.90)
        results.append(dict(label=row['label'],object=row['object'],samples=len(ix),positive_samples=int(pos.sum()),
            mean_absolute_error_BW=e.mean(0).tolist(),p99_absolute_error_BW=np.quantile(e,.99,axis=0).tolist(),
            false_load=false_load,loaded_recall=recall,checks=checks,passed=all(checks.values())))
    objects={r['object'] for r in results if r['object']!='original_flat_and_inclined'}
    coverage=len(objects)==5 and all(sum(r['positive_samples'] for r in results if r['object']==o)>=25 for o in objects)
    passed=coverage and all(r['passed'] for r in results)
    result=dict(passed=passed,whole_start_validation=True,held_out_episodes=len(results),five_furniture_positive_coverage=coverage,
        cases=results,criterion='Every held-out start: MAE<=.02BW,99th percentile error<=.05BW,final250 MAE<=.02BW,zero false lower-confidence contact triggers,>=90% loaded recall.',
        physical_success_criteria_changed=False,physical_success_claim=False,simulation_steps=0,PPO_updates=0,
        next='Admit only if every gate passes. Otherwise preserve rejection; no automatic fit retry or controller use.')
    (HERE/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(dict(passed=passed,positive_coverage=coverage,cases_passed=sum(r['passed'] for r in results),cases=len(results),worst_mean_error=max(max(r['mean_absolute_error_BW']) for r in results))),flush=True)
    if not passed:raise SystemExit(3)
if __name__=='__main__':main()
