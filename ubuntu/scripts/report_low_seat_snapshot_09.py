"""Immutable interim snapshot; deliberately not a final PPO comparison."""
import json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'progress_snapshot_09'
def read(p):return json.loads(Path(p).read_text())
def main():
    O.mkdir(exist_ok=False)
    names=['damped_original_01','clearance_damped_01','load_feedback_original_02','load_feedback_clearance_02',
      'normal_descent_03','hover_rotate_normal_03','normal_ankle_balance_04','normal_load_balance_04',
      'normal_task_mild_05','normal_task_firm_05','normal_landing2s_06','normal_landing3s_06','normal_slow_orientation_08']
    labels=['Damping','Deeper','Load','Load/deeper','Normal5s','Large lean','Ankle balance','Load balance',
      'Pose mild','Pose firm','Normal2s','Normal3s','Slower lean']
    aa=[read(D/n/'independent_audit_v2.json') for n in names]
    fig,ax=plt.subplots(figsize=(15,6),layout='constrained');x=np.arange(len(names));w=.25
    for j,(k,c,label) in enumerate([('clean','#278461','Clean'),('slipping','#d18c13','Slipping'),('unwanted_contacts','#bd3843','Unwanted contacts')]):
        ax.bar(x+(j-1)*w,[a['candidate'][k] for a in aa],w,color=c,label=label)
    ax.axhline(8,color='#278461',ls='--',label='Matched resting parent:8clean')
    ax.set(xticks=x,xticklabels=labels,ylim=(0,25),ylabel='Cases out of24',title='Thirteen physical probes: partial gains, unresolved slipping and housing contacts')
    plt.setp(ax.get_xticklabels(),rotation=25,ha='right');ax.legend(ncols=4,loc='upper center',fontsize=10)
    fig.savefig(O/'comparison.png',dpi=150);plt.close(fig)
    retention=read(D/'current_retention_07/retention.json')
    fig,ax=plt.subplots(figsize=(9,4),layout='constrained');x=np.arange(3)
    ax.bar(x-.18,[z['retained'] for z in retention.values()],.36,label='Existing successes retained',color='#278461')
    ax.bar(x+.18,[z['summary']['clean'] for z in retention.values()],.36,label='All clean cases',color='#3c6ca3')
    ax.axhline(20,color='#333',ls='--',label='Union retention requirement')
    ax.set(xticks=x,xticklabels=['Old development','Prior transfer','Recent development'],ylim=(0,25),ylabel='Cases out of24',title='Original selector retained56/60 existing successes — rejected')
    ax.legend(loc='lower left');fig.savefig(O/'retention.png',dpi=150);plt.close(fig)
    for n,a in zip(names,aa):(O/(n+'_audit.json')).write_text(json.dumps(a,indent=2))
    for n in ['timing_diagnosis_v2_06.json','housing_geometry_diagnosis_08.json','selector_revision_08.json','selector_revision_09.json',
      'retention_targets.json','reservation.json','prior_recheck_after_timing_06.json','slow_orientation_screen_08.json']:
        shutil.copy2(D/n,O/n)
    (O/'original_retention.json').write_text(json.dumps(retention,indent=2))
    rows=''.join(f'<tr><td>{n}</td><td>{a["candidate"]["stable"]}</td><td>{a["candidate"]["clean"]}</td><td>{a["candidate"]["slipping"]}</td><td>{a["candidate"]["unwanted_contacts"]}</td><td>{a["added_physical_failure_cases"]}</td></tr>' for n,a in zip(names,aa))
    (O/'findings.html').write_text('''<!doctype html><html><meta charset="utf-8"><title>Low-seat repair: interim evidence</title><style>body{font:17px system-ui;color:#172231;max-width:1250px;margin:30px auto;padding:20px}img{width:100%}table{border-collapse:collapse;width:100%}td,th{padding:8px;border-bottom:1px solid #ccc;text-align:left}.notice{padding:18px;background:#fff1d7;border-radius:8px}</style><h1>Low-seat repair is still unresolved</h1><div class="notice">No new PPO, fresh-holdout simulation or policy promotion. This is a completed-results snapshot taken before the two narrower-selector retention checks finish.</div><p>The three-second normal landing improves the difficult24-case probe from8 to14 clean cases, with22 stable cases and no added physical failures. It still has10 sliding cases and2 housing contacts. This probe uses the resting parent alone; its unchanged mild-slope slips are not regressions of the current early policy.</p><p>Combining it with the frozen early policy exposed routing regressions: nominal10° seats measured10.05°, and nominal39cm seats measured38.95cm. The original72-case comparison retained only56/60 current successes. Its failures are preserved. Separate full-bank checks are now testing measured boundaries above10.5° and below38cm, keeping the existing40cm/+20° branch. These change controller routing, not physical success criteria.</p><img src="retention.png"><p>The requirement is the union of current-policy successes:20 in each bank. Earlier holdouts are development data here; the newly reserved24-case holdout remains closed.</p><p>The slower0.55rad lean is rejected:16stable,14clean,4slipping and8housing-contact cases, including6 added physical failures. Fewer slips therefore do not establish a better controller. Recorded mesh poses bracket the actual housing penetration; this is not just a missing-geometry alarm.</p><img src="comparison.png"><table><tr><th>Candidate</th><th>Stable</th><th>Clean</th><th>Slip</th><th>Contact union</th><th>Added physical cases</th></tr>'''+rows+'''</table><p>Every probe lasts the full20seconds. Contacts are the union of forbidden-body and illegal support-patch contact; slipping and contacts can overlap. Conservative margin warnings remain separate in the audits. Exact starts/first10s and fixed-finger/bounded-controller audits passed; those audit passes do not mean physical success.</p><p>New effort gains are not established. Effort comparisons must use common complete clean trials only. The 24,242 protected prior files were rechecked intact after timing tests. The latest candidates, both routing revisions and all original failures remain preserved.</p><p>This is reference-assisted simulation, not unassisted sitting or real-robot validation. Nearby rigid-seat tests do not demonstrate sitting or leaning anywhere; leaning has not been broadened by this work.</p></html>''')
    (O/'SHA256.json').write_text(json.dumps({p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in O.iterdir() if p.is_file()},indent=2))
    print(O)
if __name__=='__main__':main()
