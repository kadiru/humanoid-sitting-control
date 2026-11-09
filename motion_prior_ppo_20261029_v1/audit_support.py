"""Independent CPU arithmetic for controlled body-collider load tests."""
import argparse
from pathlib import Path
import numpy as np
from common import ROOT,read,write,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.output)
    cases=read(ROOT/'experiments/furniture_test_20261028_v1/cases.json')
    plans=read(ROOT/'experiments/furniture_validation_20261028_v2/probe_points.json')
    results=read(root/'result.json');records=read(root/'measurements.json');assert results['complete'] and len(records)==3
    load=results['load_N'];rows=[]
    for j,round_records in enumerate(records):
        assert len(round_records)==300 and all(len(f)==14 for f in round_records)
        for i,c in enumerate(cases):
            steady=[f[i] for f in round_records if 3<=f[i]['t']<=4.5]
            measured=[];residual=[];filters=[]
            for x in steady:
                assert x['case']==c['id'] and x['round']==j
                force=np.asarray(x['force']);normal=np.asarray(x['normal_force']);friction=np.asarray(x['friction_force'])
                np.testing.assert_allclose(force,normal+friction,atol=1e-4,rtol=1e-6)
                measured.append(force@np.asarray(plans[c['id']][j]['normal']))
                residual.append(np.linalg.norm(10*np.asarray(x['acceleration'])-np.asarray(x['applied_force'])-force))
                filters.append(np.linalg.norm(normal-np.asarray(x['net_normal_force'])))
            checks=dict(sustained_contact=bool(np.min(measured)>.1*load),load_capacity=bool(.8*load<np.mean(measured)<1.2*load),
                force_balance=bool(np.median(residual)<.05*load),filter_consistency=bool(max(filters)<.05*load))
            live=next(x for x in results['checks'] if x['case']==c['id'] and x['round']==j)
            assert checks==live['checks'] and all(checks.values())==live['physical_support']
            rows.append(dict(case=c['id'],round=j,body=live['body'],physical_support=live['physical_support'],
                intended_contact_patch=live['intended_contact_patch'],mean_load_N=float(np.mean(measured)),checks=checks))
    write(root/'independent_audit.json',dict(passed=True,rows=rows,load_N=load,
        support_passes=sum(x['physical_support'] for x in rows),patch_passes=sum(x['intended_contact_patch'] for x in rows),
        scope='42 controlled body-collider fixtures, not autonomous robot successes.',
        files={str(root/n):sha(root/n) for n in ['result.json','measurements.json','manifest.json']}))

if __name__=='__main__':main()
