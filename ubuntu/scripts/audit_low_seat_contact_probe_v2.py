"""Independently summarize completed physical probes without rerunning physics."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np

def read(p):return json.loads(Path(p).read_text())
def clean(r):return bool(r['success'] and not any(r['failures'].values()) and not r['slip_ticks'] and not r['illegal_ticks'])
def unwanted(r):return bool(r['failures']['forbidden'] or r['illegal_ticks'])
def summarize(rows):
    return dict(count=len(rows),stable=sum(r['success'] for r in rows),clean=sum(map(clean,rows)),
        slipping=sum(r['slip_ticks']>0 for r in rows),unwanted_contacts=sum(map(unwanted,rows)),
        physical_failures=sum(any(r['failures'].values()) for r in rows),
        margin_warnings=sum(r['traction_bad_ticks']>0 for r in rows),
        full_duration=all(r['steps']==1000 for r in rows))

def main():
    p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,required=True);a=p.parse_args()
    base=a.folder/'baseline';trial=a.folder/'candidate';br=read(base/'results.json')['episodes'];cr=read(trial/'results.json')['episodes']
    assert len(br)==len(cr)
    exact={}
    for name in ['startup.npz','traction_startup.npz','bridge_startup.npz']:
        with np.load(base/name) as b,np.load(trial/name) as c:
            exact[name]={k:bool(np.array_equal(b[k],c[k])) for k in b.files}
    prefix={}
    with np.load(base/'trace.npz') as b,np.load(trial/'trace.npz') as c:
        assert np.array_equal(b['t'],c['t'])
        for k in ['body','q','command','reference','geometry']:
            mask=b['t']<=10.
            prefix[k]=dict(exact=bool(np.array_equal(b[k][mask],c[k][mask])),max_error=float(abs(b[k][mask]-c[k][mask]).max()))
    lost=[i for i,(b,c) in enumerate(zip(br,cr)) if clean(b) and not clean(c)]
    gained=[i for i,(b,c) in enumerate(zip(br,cr)) if not clean(b) and clean(c)]
    added_physical=[i for i,(b,c) in enumerate(zip(br,cr)) if not any(b['failures'].values()) and any(c['failures'].values())]
    lost_stable=[i for i,(b,c) in enumerate(zip(br,cr)) if b['success'] and not c['success']]
    common=[i for i,(b,c) in enumerate(zip(br,cr)) if clean(b) and clean(c)]
    pairs={}
    for window in ['metrics_whole_trial','metrics_last10s','metrics_last5s']:
        if not common:continue
        pairs[window]={}
        for key in ['total_squared_torque_Nm2','total_absolute_torque_Nm','all_joint_rms_Nm','leg_squared_torque_Nm2','upper_squared_torque_Nm2']:
            if window not in br[0]:continue
            b=np.array([br[i][window][key] for i in common]);c=np.array([cr[i][window][key] for i in common])
            pairs[window][key]=dict(n=len(common),baseline=float(b.mean()),candidate=float(c.mean()),percent_change=float((c.mean()/b.mean()-1)*100))
    output=dict(baseline=summarize(br),candidate=summarize(cr),lost_clean_cases=lost,gained_clean_cases=gained,
        added_physical_failure_cases=added_physical,lost_stable_cases=lost_stable,
        common_clean_effort=pairs,exact_startups=exact,locked_first10=prefix,
        rows=[dict(case=i,baseline_clean=clean(b),candidate_clean=clean(c),stable=c['success'],
            slip_ticks=c['slip_ticks'],illegal_ticks=c['illegal_ticks'],failures=c['failures']) for i,(b,c) in enumerate(zip(br,cr))])
    output['regression_gate_passed']=not lost and not added_physical and not lost_stable and all(all(x.values()) for x in exact.values()) and all(x['exact'] for x in prefix.values())
    output['repair_complete']=output['regression_gate_passed'] and output['candidate']['clean']==len(cr)
    dest=a.folder/'independent_audit_v2.json';assert not dest.exists();dest.write_text(json.dumps(output,indent=2))
    print(json.dumps({k:v for k,v in output.items() if k not in ['rows','common_clean_effort']},indent=2))

if __name__=='__main__':main()
