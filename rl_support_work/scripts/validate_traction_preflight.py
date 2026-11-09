"""Physical admission and exact reference regression; margins remain visible."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from verify_seat_load_study import evaluation,starts,load,equal
from verify_progressive_seat_study import quality
DATA=Path('/data/humanoid-sitting-perception/traction_transfer_01')
def read(p):return json.loads(Path(p).read_text())
def physical(path):
    summary,rows,g=evaluation(path,18);quality(summary,rows)
    assert summary['successes']==18 and summary['physical_failures']==0
    assert all(e['slip_ticks']==0 and e['illegal_ticks']==0 for e in rows)
    z=np.load(path/'every_tick.npz');assert z['force'].shape==(1000,18,7,3)
    f=z['force'];n=z['normal'];fn=(f*n).sum(-1);ft=np.linalg.norm(f-fn[...,None]*n,axis=-1);active=np.linalg.norm(f,axis=-1)>.01*z['weight'][...,None]
    utilization=np.where(active,ft/(.6*np.maximum(fn,1e-7)),0.)
    maximum=utilization.max((0,2));assert maximum.max()<=1.,maximum
    assert z['no_slip'].all() and not z['failure'].any() and (z['wrong']<=10).all()
    counts=(~z['traction']).sum(0);assert counts.tolist()==[e['traction_bad_ticks'] for e in rows]
    return dict(**summary,all_tick_friction_max=maximum.tolist(),friction_margin_ticks=counts.tolist())
def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--with-smokes',action='store_true');a=p.parse_args();out=Path(a.output);assert not out.exists()
    candidate=Path(read(DATA/'selected_controller.json')['validated_probe']);ref=DATA/'env_reference_01'
    baseline=physical(candidate);reference=physical(ref);assert baseline==reference
    start=starts(candidate,ref);assert start['exact']
    x=np.load(candidate/'trace.npz');y=np.load(ref/'trace.npz');assert x.files==y.files
    for k in x.files:assert np.array_equal(x[k],y[k]),k
    reset=read(DATA/'env_reset_01/reset_probe.json');assert reset['passed']
    output=dict(passed=True,physical=reference,reference_probe_all_trace_arrays_exact=True,startups_exact=True,reset=reset,smokes={},scope='Stable/slip/contact admission requires all18. Original0.90 conservative friction margin and strict-success labels are unchanged; actual utilization must stay<=1.0.')
    if a.with_smokes:
        for seed in [58311,58317]:
            train=DATA/f'smoke_{seed}';evaluation_path=DATA/f'smoke_eval_{seed}'
            smoke=physical(evaluation_path);assert np.all(np.array(smoke['friction_margin_ticks'])<=np.array(reference['friction_margin_ticks']))
            initial=load(train/'initial.pt');final=load(train/'final.pt');loaded=load(evaluation_path/'initial.pt')
            for role in ['actor_state_dict','critic_state_dict']:
                assert equal(final[role],loaded[role]);assert any(not equal(initial[role][k],v) for k,v in final[role].items())
            v=read(train/'verification.json');assert v['updates']==8 and v['transitions']==23040
            assert starts(ref,evaluation_path)['exact']
            s=np.load(ref/'traction_startup.npz');t=np.load(evaluation_path/'traction_startup.npz');assert all(np.array_equal(s[k],t[k]) for k in s.files)
            x=np.load(ref/'trace.npz');y=np.load(evaluation_path/'trace.npz');keep=x['t']<=11
            for k in ['q','body','command','torque','reference','geometry']:assert np.array_equal(x[k][keep],y[k][keep]),(seed,k)
            output['smokes'][str(seed)]=dict(**smoke,verification=v,checkpoint_sha256=hashlib.sha256((train/'final.pt').read_bytes()).hexdigest(),actor_and_critic_changed=True,exact_checkpoint_loaded=True,teacher_prefix_exact_through_s=11)
    out.write_text(json.dumps(output,indent=2));print(json.dumps(output,indent=2))
if __name__=='__main__':main()
