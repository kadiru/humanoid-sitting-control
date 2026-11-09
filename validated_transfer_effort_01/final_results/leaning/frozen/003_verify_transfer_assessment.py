"""Read-only assessment verification using the established independent physical checker."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
from verify_split_leaning import evaluation,starts,read,sha

def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    a.output.mkdir(exist_ok=False);shutil.copy2(__file__,a.output/'verification_source.py')
    root=a.data.parent.parent;old=read(root/'prior_preservation.json')['hashes']
    for f,h in old.items():assert sha(f)==h,f
    assert read(a.data/'identity_verification.json')['passed']
    assert read(a.data/'reset_isolation/reset_probe.json')['passed']
    report={'passed':False,'protected_hashes':len(old),'identity_arrays_exact':True,'reset_isolation':True,'evaluations':{},'startups':{},'by_geometry':{}}
    labels=['fixed','reference']+[x.name for x in a.data.glob('policy_*') if x.is_dir()]
    for label in labels:
        d=a.data/label;m=read(d/'manifest.json');check,rows=evaluation(d,18,Path(m['args']['checkpoint']) if m['args']['checkpoint'] else None)
        report['evaluations'][label]=check;report['startups'][label]=starts(a.data/'fixed',d)
        geoms=read(d/'geometry.json');summary=[]
        for j,g in enumerate(geoms):
            cases=[x for x in rows['episodes'] if x['geometry_index']==j]
            summary.append(dict(label=g['label'],height=g['height'],slope=g['pitch_deg'],known_anchor=g['known_anchor'],count=len(cases),success=sum(x['success'] for x in cases),physical_failures=sum(any(x['failures'].values()) for x in cases),failed_flags={k:sum(x['failures'][k] for x in cases) for k in cases[0]['failures']},final_hold_s=[x['final_hold_s'] for x in cases]))
        report['by_geometry'][label]=summary
    report['passed']=True
    (a.output/'verification.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    print(json.dumps({'passed':True,'results':{k:dict(success=v['successes'],physical_failures=v['physical_failures']) for k,v in report['evaluations'].items()}},indent=2))
if __name__=='__main__':main()
