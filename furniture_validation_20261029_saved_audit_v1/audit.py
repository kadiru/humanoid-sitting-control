"""CPU-only audit of already saved probes; never changes the physical criteria."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--source',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);ap.add_argument('--points',type=Path,required=True);a=ap.parse_args()
    points=json.loads(a.points.read_text())
    records=json.loads((a.source/'measurements.json').read_text())
    result=json.loads((a.source/'result.json').read_text())
    assert result['complete'] and result['probe_steps']==4800
    assert len(records)==4 and all(len(r)==300 for r in records)
    differences=[];checks=[]
    for ri,round_frames in enumerate(records):
        for ci in range(14):
            frames=[frame[ci] for frame in round_frames]
            for x in frames:
                a32,u32,n32,t32=(np.asarray(x[k],np.float32) for k in ['acceleration','applied_force','normal_force','friction_force'])
                # Reproduce the logger's subtraction order and arithmetic dtype.
                logged=float(np.linalg.norm(((a32-u32)-n32)-t32))
                np.testing.assert_allclose(logged,x['balance_residual_N'],atol=2e-5,rtol=2e-5)
                f=np.asarray(x['normal_force'],float)+np.asarray(x['friction_force'],float)
                np.testing.assert_allclose(f,x['force'],atol=2e-6,rtol=2e-6)
                direct=float(np.linalg.norm(a32.astype(float)-u32.astype(float)-f))
                differences.append(abs(direct-logged))
                assert np.isfinite(np.r_[f,a32,u32,x['position'],x['velocity']]).all()
            steady=[x for x in frames if 3<=x['t']<=4.5]
            balances=[float(np.linalg.norm(np.asarray(x['acceleration'])-x['applied_force']-np.asarray(x['normal_force'])-x['friction_force'])) for x in steady]
            row=next(x for x in result['checks'] if x['case']==frames[0]['case'] and x['round']==ri)
            if ri<3:
                # Independent high-precision balance remains subject to the original 1 N limit.
                assert bool(np.median(balances)<1)==row['checks']['force_balance']
                assert (max(x['filter_error_N'] for x in steady)<.1)==row['checks']['filters']
                assert all(x['labels'] and all(l['valid'] for l in x['labels']) for x in steady)==row['checks']['target_mapping']
                point=np.asarray(points[row['case']][ri]['point']);normal=np.asarray(points[row['case']][ri]['normal'])
                loads=np.asarray([np.dot(x['force'],normal) for x in steady])
                assert bool((loads>1).all())==row['checks']['sustained_contact']
                assert bool(8<loads.mean()<12)==row['checks']['load_magnitude']
                assert bool(min((np.asarray(x['position'])-point)@normal for x in steady)>=.01)==row['checks']['no_through_surface']
                mechanical=all(v for k,v in row['checks'].items() if k!='target_mapping')
                checks.append(dict(case=row['case'],round=ri,mechanical=mechanical,target_mapping=row['checks']['target_mapping'],median_force_balance_error_N=float(np.median(balances)),mean_normal_force_N=row['mean_normal_force_N']))
            else:
                assert max(np.linalg.norm(x['force']) for x in steady)<.1
    a.output.mkdir(exist_ok=True,parents=True)
    report=dict(passed=True,scope='Saved low-load probe arithmetic and physical outcome audit; no simulation replay. Each recorded mechanical outcome is independently recomputed. Contact region labels are inspected as saved, not a new geometric mapping algorithm.',frames=16800,
        explanation='Original audit regrouped float32 normal/friction subtraction after conversion to float64. This audit reproduces the logged dtype and subtraction order with the original arithmetic tolerance, and independently checks float64 force balance against the unchanged 1 N physical limit.',
        maximum_logged_vs_float64_residual_difference_N=max(differences),mechanical_passes=sum(x['mechanical'] for x in checks),surface_presses=42,empty_space_controls=14,target_mapping_passes=sum(x['target_mapping'] for x in checks),fully_validated_layouts=result['validated_layouts'],checks=checks,
        whole_robot_success_claimed=False,inputs={f:hashlib.sha256((a.source/f).read_bytes()).hexdigest() for f in ['measurements.json','result.json']})
    (a.output/'audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print({k:v for k,v in report.items() if k not in ['checks','inputs','scope']})

if __name__=='__main__':main()
