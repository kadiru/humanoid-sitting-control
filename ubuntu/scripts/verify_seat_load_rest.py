"""Independently audit the new rest-only target, separate from transition claims."""
import argparse,json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main():
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();a.output.mkdir(exist_ok=False)
    r=a.root;rows=json.loads((r/'rest_replay_02/results.json').read_text())['rows'];t=np.load(r/'rest_replay_02/trace.npz')
    assert len(rows)==18 and all(x['valid'] and x['steps']==1000 and x['good_fraction']==1 and x['final_hold_seconds']==20 and not any(x['failures'].values()) for x in rows)
    assert t['t'][0]==.1 and t['t'][-1]==20 and len(t['t'])==200
    bank=np.load(r/'rest_replay_bank_01/candidates.npz');end=np.load(r/'rest_replay_02/candidates.npz');assert np.array_equal(bank['command'].astype(np.float32),end['command']);assert np.all(t['command']==bank['command'].astype(np.float32)[None])
    torque=t['torque'];total=(torque**2).sum(-1);load=t['vertical_support'];foot=t['foot']
    assert np.isfinite(torque).all() and (load[-50:].mean(0)>.5).all()
    before=json.loads((r/'prior_preservation.json').read_text());hashes=before.get('hashes',before)
    for name,h in hashes.items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==h,name
    result=dict(passed=True,prior_hashes_verified=len(hashes),trials=18,unique_surfaces=9,seconds_per_trial=20,physical_failures=0,constant_commands=True,all_ticks_good=True,
        last5_total_squared_torque_Nm2=total[-50:].mean(0).tolist(),last5_all_joint_rms_Nm=np.sqrt(total[-50:]/37).mean(0).tolist(),last5_seat_vertical_load_bw=load[-50:].mean(0).tolist(),last5_foot_load_bw=foot[-50:].mean(0).tolist(),
        scope='Already-seated reset/retention test. Does not demonstrate standing transitions, unseen-surface success, electrical energy reduction or a global effort minimum.')
    (a.output/'verification.json').write_text(json.dumps(result,indent=2))
    old=json.loads((r/'upright_transition_01/results.json').read_text())['episodes'][:9]
    ref=np.array([x['metrics_last5s']['total_squared_torque_Nm2'] for x in old]);refl=np.array([x['metrics_last5s']['vertical_support_load_bw'] for x in old]);new=total[-50:,:9].mean(0);newl=load[-50:,:9].mean(0)
    labels=[f"{x['geometry']['height']*100:.0f}cm\n{x['geometry']['pitch_deg']:+.0f}°" for x in rows[:9]]
    fig,axs=plt.subplots(1,2,figsize=(12,4.6));xx=np.arange(9)
    axs[0].bar(xx-.18,ref,.36,label='Current reference: final5s after standing');axs[0].bar(xx+.18,new,.36,label='New seated-reset target: final5s')
    axs[0].set_ylabel('Sum of squared joint torque (Nm²)');axs[0].set_title('Resting effort — different starting states')
    axs[1].bar(xx-.18,100*refl,.36);axs[1].bar(xx+.18,100*newl,.36);axs[1].set_ylabel('Weight carried by seat (%)');axs[1].set_title('Seat carries most of the weight');axs[1].set_ylim(0,100)
    for ax in axs:ax.set_xticks(xx,labels);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.legend(loc='upper center',bbox_to_anchor=(.5,.95),ncol=2,frameon=False);fig.suptitle('Validated seated target:18/18 full20-second trials; zero physical failures',y=1.02);fig.tight_layout(rect=(0,0,1,.88));fig.savefig(a.output/'rest_validation.png',dpi=160,bbox_inches='tight');plt.close(fig)
    shutil.copy2(__file__,a.output/'verifier_source.py')
    files=[x for x in a.output.iterdir() if x.is_file()];(a.output/'SHA256.json').write_text(json.dumps({x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in files},indent=2));print(json.dumps(result))

if __name__=='__main__':main()
