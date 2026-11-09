import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1];out=root/'runs/soft_landing_01'
conditions=[('Confirmation clean','deployment_status_fresh_05','soft_landing_no_contact_clean_01','soft_landing_actions_clean_01'),('Confirmation noisy','deployment_status_fresh_noisy_05','soft_landing_limiter_confirmation_noise_01','soft_landing_actions_noisy_01'),('Fresh furniture clean','soft_landing_baseline_fresh_clean_01','soft_landing_limiter_fresh_clean_01','soft_landing_actions_fresh_clean_01'),('Fresh furniture noisy','soft_landing_baseline_fresh_noisy_01','soft_landing_limiter_fresh_noise_01','soft_landing_actions_fresh_noisy_01')]
fig,axs=plt.subplots(2,2,figsize=(12,8),sharex=True,sharey=True)
for ax,(title,*names) in zip(axs.flat,conditions):
 for name,label,color in zip(names,['Original student','Fixed limiter','Contact feedback'],['gray','tab:blue','tab:green']):
  r=json.loads((root/'runs'/name/'results.json').read_text());force=np.sort([o['max_seat_impact_N'] for o in r['outcomes']]);ax.plot(force,np.arange(1,len(force)+1)/len(force),label=label,color=color)
 ax.axvline(632.527832,color='tab:red',ls='--');ax.set_title(title);ax.grid(alpha=.2);ax.set_xlabel('Peak seat force per attempt (N)');ax.set_ylabel('Fraction of attempts');ax.legend(fontsize=9)
fig.suptitle('Matched initial conditions; all attempts including safety failures');fig.tight_layout();fig.savefig(out/'impact_distributions.png',dpi=160);plt.close(fig)
fig,axs=plt.subplots(4,1,figsize=(11,10),sharex=True)
for name,label,color in [('soft_landing_baseline_video_01','Original student','gray'),('soft_landing_contact_video_01','Contact-aware limiter','tab:green')]:
 z=np.load(root/'runs'/name/'rollouts.npz');t=(np.arange(len(z['support']))+1)*.02
 axs[0].plot(t,z['root_pose'][:,0,2],label=label,color=color)
 axs[1].plot(t,z['seat_force_peak'][:,0],label=label,color=color)
 speed=np.abs(np.diff(z['targets'][:,0,15:17],axis=0)/.02).max(-1);axs[2].plot(t[1:],speed,label=label,color=color)
 if 'estimated_contact' in z:
  axs[3].plot(t,z['estimated_contact'][:,0,0],label='Predicted contact probability',color='tab:orange');axs[3].plot(t,z['estimated_contact'][:,0,1],label='Estimated load / mg',color='tab:green');axs[3].plot(t,z['support'][:,0].clip(0,1),label='Simulation load label (diagnostic)',color='tab:blue')
axs[0].set_ylabel('Pelvis height (m)');axs[1].set_ylabel('5 ms peak force (N)');axs[1].axhline(632.527832,color='r',ls='--');axs[2].set_ylabel('Ankle target speed (rad/s)');axs[3].set_ylabel('Contact / normalized load');axs[3].set_xlabel('Time (s)')
for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Paired video: 973 N force failure versus 512 N safe transition');fig.tight_layout();fig.savefig(out/'paired_physics.png',dpi=160);plt.close(fig)
