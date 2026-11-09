"""Freeze the admitted setup; no training starts before this succeeds."""
import json,time
from pathlib import Path
from posture_credit_common import *
from run_posture_credit_preflight import physical
from verify_seat_load_study import load,equal,evaluation,starts
from run_transfer_assessment import write
import torch,numpy as np
from verify_posture_credit_study import replay_optimizer

def main():
 pre=DATA/'preflight';study=DATA/'ppo_01';assert not study.exists()
 assert read(pre/'zero_action_validation.json')['passed']
 admission={}
 for gpu,seed in enumerate(SEEDS):
  assert read(DATA/f'preflight_status_gpu{gpu}.json')['phase']=='complete'
  admission[str(seed)]={}
  for label,count in [(f'explore_{seed}',3),(f'smoke_{seed}',2),(f'smoke_eval_{seed}',1)]:admission[str(seed)][label]=physical(pre/label,count)
  cp=pre/f'smoke_{seed}/final.pt';evaluation(pre/f'smoke_eval_{seed}',18,cp)
  assert starts(pre/'calibrate',pre/f'smoke_eval_{seed}')['exact']
  x=np.load(pre/'calibrate/trace.npz');y=np.load(pre/f'smoke_eval_{seed}/trace.npz');mask=x['t']<=11
  for k in ['q','body','command','reference','torque','geometry']:assert np.array_equal(x[k][mask],y[k][mask]),(seed,k)
  initial=load(pre/f'smoke_{seed}/initial.pt');final=load(cp)
  assert final['updates']==2
  replay_optimizer(pre/f'smoke_{seed}',read(pre/'calibrate/episode_calibration.json'),updates=2)
  for role in ['actor_state_dict','critic_state_dict']:
   assert all(torch.isfinite(x).all() for x in final[role].values())
   assert any(not torch.equal(x,final[role][k]) for k,x in initial[role].items())
 prior=read(DATA/'prior_hashes.json');check_hashes(prior)
 for p in pre.rglob('*'):
  if p.is_file():prior[str(p)]=sha(p)
 for p in (DATA/'probes').rglob('*'):
  if p.is_file():prior[str(p)]=sha(p)
 for name in ['diagnosis','development_archive_01']:
  for p in (DATA/name).rglob('*'):
   if p.is_file():prior[str(p)]=sha(p)
 old=read(OLD/'ppo_01/protocol.json');hashes=dict(old['hashes'])
 names=['run_posture_episode_ppo.py','posture_episode_ppo.py','posture_credit_common.py','run_posture_credit_study.py','run_posture_credit_preflight.py','freeze_posture_credit_study.py','test_posture_episode_ppo.py','verify_posture_credit_study.py','report_posture_credit_study.py','render_posture_credit_replay.py','finalize_posture_credit_study.py']
 files=[ROOT/'scripts'/n for n in names]+[ROOT/'docs/POSTURE_CREDIT_PROTOCOL.md',pre/'calibrate/episode_calibration.json',DATA/'holdout_reservation_01.json']+list((DATA/'reserved_bank_01').iterdir())
 for p in files:hashes[str(p)]=sha(p)
 protocol=dict(admitted=True,utc_timestamp=time.time(),seeds=SEEDS,updates=10,active_per_seed=180000,warmup_per_seed=4050,transitions_per_seed=184050,total_transitions=368100,posture_decisions_per_seed=180,initial_settling_total_excluded=900,
  source_files=names,hashes=hashes,prior_hashes=prior,preflight=admission,zero_action_regression=read(pre/'zero_action_validation.json'),
  development_baselines={'reference':str(pre/'calibrate'),'previous_58311':str(OLD/'ppo_01/seed_58311/development_final'),'previous_58317':str(OLD/'ppo_01/seed_58317/development_final')},
  development_only_calibration=str(pre/'calibrate/episode_calibration.json'),holdout=str(DATA/'reserved_bank_01'),
  contract='414 sensor/reference inputs; three bounded symmetric offsets selected once11s, held until20s; no live truth actor/critic inputs',
  caveats=['Bundled learning change, not a single-factor ablation','Two seeds; nearby rigid surfaces repeated twice','Reference controls the first11s and supplies closed-loop feedback afterward','Torque proxy, not energy/global optimum','Original conservative prefix margin flags remain separately reported'])
 study.mkdir();write(study/'protocol.json',protocol);(study/'protocol.sha256').write_text(sha(study/'protocol.json')+'\n')
 print(json.dumps(dict(admitted=True,source_hashes=len(hashes),prior_hashes=len(prior),transitions=368100)))
if __name__=='__main__':main()
