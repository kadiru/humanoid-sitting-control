"""Independent saved-record force balance, force filter and scope audit."""
import numpy as np
from io_utils import OUT,ASSETS,HERE,read,write,sha

def main():
    result=read(OUT/'probes/result.json');records=read(OUT/'probes/measurements.json')
    cases=read(ASSETS/'cases.json');points=read(HERE/'probe_points.json')
    assert result['complete'] and len(records)==4 and all(len(r)==300 for r in records)
    assert result['probe_steps']==4800 and result['simulation_steps']<=5800
    validated=[]
    for ci,c in enumerate(cases):
        case_pass=True
        for ri in range(4):
            frames=[r[ci] for r in records[ri]]
            assert len(frames)==300 and all(x['case']==c['id'] and x['round']==ri for x in frames)
            for x in frames:
                f=np.array(x['force']);a=np.array(x['acceleration']);u=np.array(x['applied_force'])
                assert np.isfinite(np.r_[f,a,u,x['position'],x['velocity']]).all()
                np.testing.assert_allclose(f,np.array(x['normal_force'])+x['friction_force'],atol=2e-6,rtol=2e-6)
                np.testing.assert_allclose(np.linalg.norm(a-u-f),x['balance_residual_N'],atol=2e-5,rtol=2e-5)
            steady=[x for x in frames if 3<=x['t']<=4.5]
            if ri<3:
                n=np.array(points[c['id']][ri]['normal']);p=np.array(points[c['id']][ri]['point'])
                loads=np.array([np.dot(x['force'],n) for x in steady])
                checks=[(loads>1).all(),8<loads.mean()<12,
                    np.median([np.linalg.norm(np.array(x['acceleration'])-x['applied_force']-np.array(x['force'])) for x in steady])<1,
                    max(x['filter_error_N'] for x in steady)<.1,
                    min((np.array(x['position'])-p)@n for x in steady)>=.01,
                    all(x['labels'] and all(l['valid'] for l in x['labels']) for x in steady)]
                passed=bool(all(checks))
            else:passed=bool(max(np.linalg.norm(x['force']) for x in steady)<.1)
            row=next(r for r in result['checks'] if r['case']==c['id'] and r['round']==ri)
            assert passed==row['passed'],('PROBE_AUDIT_DISAGREEMENT',c['id'],ri)
            case_pass &= passed
        if case_pass:validated.append(c['id'])
    assert validated==result['validated_layouts']
    write(OUT/'probes/independent_audit.json',dict(passed=True,validated_layouts=validated,
        robot_success_claimed=False,measured_frames=16800,result_sha256=sha(OUT/'probes/result.json'),
        measurements_sha256=sha(OUT/'probes/measurements.json')))

if __name__=='__main__':main()
