"""Summarize all outcomes without changing physical acceptance."""
from paths import ASSETS,OUT,PRIOR,PPO,read,write

def main():
    selection=read(OUT/'policy_selection.json');selected=selection['selected_original_result']
    protected=read(PRIOR/'protected_sitting_cases.json')
    rows=read(OUT/'retention72/results.json')['episodes'];clean={x['id'] for x in rows if x['clean']}
    oldrows=read(PPO/selected['label']/'results.json')['episodes'];oldclean={x['id'] for x in oldrows if x['clean']}
    retention=dict(clean=len(clean),repairs=sorted(clean&set(protected['failed'])),regressions=sorted(set(protected['protected'])-clean),
        lost_vs_same_checkpoint_original_perception=sorted(oldclean-clean),gained_vs_same_checkpoint_original_perception=sorted(clean-oldclean),
        independent_audit=read(OUT/'retention72/independent_audit.json')['passed'])
    cases=[]
    for c in read(ASSETS/'cases.json'):
        folder=OUT/('robot_'+c['id']);depth=read(folder/'depth_audit.json');robot=read(folder/'independent_audit.json')
        cases.append(dict(id=c['id'],task=c['task'],approach=c['approach'],depth=depth,robot=robot))
    body=read(OUT/'body_support/independent_audit.json') if (OUT/'body_support/independent_audit.json').exists() else None
    write(OUT/'summary.json',dict(complete=True,retention72=retention,furniture=cases,body_support=body,
        physical_thresholds_changed=False,action_limits_changed=False,automatic_promotion=False,
        acceptance='Best09 remains until a replacement retains all64 and repairs the failures; furniture diagnostics never count toward72.',
        curved_patch_caution='Original5deg/15mm single-plane gate preserved; exact selected-point membership is additional diagnostic evidence.'))
if __name__=='__main__':main()
