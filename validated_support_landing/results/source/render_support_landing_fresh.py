"""Select a documented validation example, never filter quantitative scores."""
import json,os,subprocess,sys
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1];store=Path('/data/humanoid-sitting-perception')
cap=store/'support_landing_fresh_capture_01';score=json.loads((cap/'shadow_score/score.json').read_text())
with np.load(cap/'depth.npz') as f:d={k:f[k] for k in ['case','t','legacy']}
counts={int(c):int(((d['case']==int(c))&(d['t']>=v['t']-1)&(d['t']<=v['t']+.5)&(d['legacy'][:,8]==0)).sum()) for c,v in score['event_cases'].items()}
rows=json.loads((cap/'results.json').read_text())['episodes']
good=[i for i in sorted(counts) if rows[i]['success']]
bad=[i for i in sorted(counts) if not rows[i]['success']]
choices={'tracked_landing':next((i for i in good if counts[i]>0),good[0]),
         'fall_failure':next((i for i in bad if rows[i]['failures']['fall']),bad[0])}
base=store/'support_landing_fresh_visual_02';base.mkdir(exist_ok=False)
for name,case in choices.items():
    t=score['event_cases'][str(case)]['t'];start=max(0.,t-2.);end=min(20.,t+3.);out=base/name
    subprocess.run([sys.executable,str(root/'scripts/render_support_landing.py'),'--tracking',str(cap/'shadow_score'),'--case',str(case),'--start',str(start),'--end',str(end),'--output',str(out)],check=True)
    p=out/'selection.json';selection=json.loads(p.read_text());selection.update(selection_rule='First successful case with original-estimator rejection, else first success; paired with first fall case. Both outcomes shown; all cases remain in scores.',original_rejections_by_case=counts)
    p.write_text(json.dumps(selection,indent=2)+'\n');print(json.dumps(selection,indent=2))
