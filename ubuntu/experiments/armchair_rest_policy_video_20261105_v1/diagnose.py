"""Read-only analysis of first failure and learned residuals; no simulation."""
import json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/kadir/humanoid-sitting');RUN=ROOT/'outputs/armchair_rest_ppo_20261105_v1';OUT=ROOT/'outputs/armchair_rest_policy_video_20261105_v1'
read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=True);folder=RUN/'learned_armchair11';meta=read(folder/'manifest.json')['metadata'];lo=np.array(meta['lower']);hi=np.array(meta['upper'])
    audit=read(folder/'independent_audit.json');assert audit['passed'];assert sha(folder/'trajectory.npz')==audit['files']['trajectory.npz']
    paths=[folder/'trajectory.npz',folder/'manifest.json',folder/'initial_selection.json',RUN/'training_65011/final.pt',Path(__file__)]
    (OUT/'diagnosis_inputs.json').write_text(json.dumps({str(p):sha(p) for p in paths},indent=2)+'\n')
    ck=torch.load(RUN/'training_65011/final.pt',map_location='cpu',weights_only=False);choice=read(folder/'initial_selection.json')['library_indices']
    with np.load(folder/'trajectory.npz') as z:
        prior=ck['library']['desired'].numpy()[choice].transpose(1,0,2);residual=z['desired']-prior;rows=[]
        for i,r in enumerate(read(folder/'results.json')['episodes']):
            hits=np.flatnonzero(z['failure'][:,i]);t=int(hits[0]) if len(hits) else 999
            q=z['q'][t,i,:23];margin=np.minimum(q-lo,hi-q);j=int(margin.argmin());maxr=np.abs(residual[:t+1,i]).max(0);k=int(maxr.argmax())
            rows.append(dict(id=r['id'],clean=r['clean'],final_hold_s=r['final_hold_s'],first_failure_s=float((t+1)*.02) if len(hits) else None,first_failure_flags=z['flags'][t,i].tolist() if len(hits) else None,minimum_margin_joint=meta['joint_names'][j],actual_joint_angle_rad=float(q[j]),actual_margin_rad=float(margin[j]),required_actual_margin_rad=.005,commanded_angle_rad=float(z['command'][t,i,j]),support_load_bw_at_sample=float(z['support_bw'][t,i]),largest_residual_joint=meta['joint_names'][k],max_abs_learned_residual_rad=float(maxr[k]),all_command_increments_within_original_limit=bool(np.max(np.abs(z['command'][:t+1,i,:23]-z['previous'][:t+1,i,:23]))<=.0030002)))
    result=dict(observations=rows,verified='9/11 learned armchair trials first failed the actual joint-margin predicate before intended support, despite respecting commanded rate and range limits. Prior11/11 had passed. No successful replacement was accepted.',inference='The learned corrections destabilize the approach. Saved-demo imitation and correctly ranked rewards did not prevent loss of the successful behavior when the policy changed its own state distribution. This does not demonstrate a local minimum or an impossible action range.',next_step='Keep the11/11 prior. Preserve its approach during future optimization; add recovery data around the observed ankle/balance deviations and screen deterministic checkpoints for retention. For straighter pelvis rest, first construct a collision-clear whole-body approach that avoids the upper torso seam, then demonstrate it before imitation/PPO.',limitations='This is saved-data analysis, not a tested repair. No additional training or simulation was run.')
    (OUT/'regression_diagnosis.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(rows,indent=2))
if __name__=='__main__':main()
