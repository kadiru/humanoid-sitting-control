"""Isolated actuator-distribution diagnostic, not a robot success benchmark."""
import json
from pathlib import Path
import shutil
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from humanoid_sitting.support_pace import update_pace
root=Path(__file__).resolve().parents[1]
out=Path('/data/humanoid-sitting-perception/support_rl_pace_results_01/pace_distribution')
out.mkdir(exist_ok=False);shutil.copy2(__file__,out/'diagnostic_source.py')
torch.set_num_threads(1);g=torch.Generator().manual_seed(51691)
n=256;rate=torch.ones(n);neutral=rate.clone();ready=torch.ones(n,dtype=torch.bool);samples=[]
with torch.inference_mode():
 for t in range(2000):
  rate=update_pace(rate,.5*torch.randn(n,generator=g),ready)
  neutral=update_pace(neutral,torch.zeros(n),ready)
  if t>=250:samples.append(rate.clone())
x=torch.stack(samples).numpy()
assert np.allclose(neutral.numpy(),1.)
r=dict(scope='Isolated action mapping with constant zero Gaussian mean and std 0.5; no robot simulation or success score.',
 seed=51691,parallel_filters=n,steps=2000,discarded_transient_steps=250,
 stochastic_filtered_pace_mean=float(x.mean()),stochastic_filtered_pace_std=float(x.std()),
 stochastic_pace_5th_percentile=float(np.quantile(x,.05)),stochastic_pace_95th_percentile=float(np.quantile(x,.95)),
 deterministic_mean_action_pace=float(neutral.mean()),
 conclusion='Upper clipping at nominal speed makes zero-mean exploration systematically slower. Executing the raw Gaussian mean is not equivalent to the average transformed training action.',
 causal_limit='This establishes an action-distribution mismatch, not that it alone caused the robot regressions. Actual policy means and states change during training.')
(out/'results.json').write_text(json.dumps(r,indent=2)+'\n')
fig,ax=plt.subplots(figsize=(8,4.5),constrained_layout=True)
ax.hist(x.ravel(),bins=70,density=True,color='#c2410c',alpha=.8,label='Random training actions after filter')
ax.axvline(1.,color='#047857',lw=2,label='Execute the Gaussian mean')
ax.set(xlim=(.65,1.025),xlabel='Motion pace setting',ylabel='Density',title='Same zero-mean policy, different effective pace\nIsolated actuator diagnostic; no robot simulation')
ax.legend();ax.grid(alpha=.2);fig.savefig(out/'distribution.png',dpi=150);plt.close(fig)
(out/'report.md').write_text('# Pace exploration mismatch\n\n'+r['scope']+'\n\n'+
 f"With a zero action mean and standard deviation 0.5, the actual pace filter averages {x.mean():.4f} under sampled training actions, versus 1.0000 when executing the mean. "+r['conclusion']+'\n\n'+r['causal_limit']+'\n')
with (out.parent/'report.md').open('a') as f:
 f.write('\n## Exploration-distribution diagnostic\n\n'+
  f"An isolated test of the implemented filter gives mean pace {x.mean():.4f} under zero-mean Gaussian exploration (std 0.5), versus 1.0000 for deterministic mean actions. "+
  'This asymmetric clipping creates a training/execution mismatch. It is a plausible contributor requiring controlled retraining, not a proven sole cause. See [the diagnostic](pace_distribution/report.md).\n')
print(json.dumps(r,indent=2))
