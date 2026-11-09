"""Whole-trial distributions including safety failures; contact labels diagnostic only."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--runs',nargs='+',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
rows=[];fig,axes=plt.subplots(2,1,figsize=(10,8))
for folder in a.runs:
 root=Path(folder);r=json.loads((root/'results.json').read_text());z=np.load(root/'rollouts.npz');peaks=z['seat_force_peak'].max(0)
 row={'run':folder,'n':r['num_trials'],'successes':sum(o['success'] for o in r['outcomes']),'falls':sum(o['fall'] for o in r['outcomes']),'unintended_contacts':sum(o['unintended_contact'] for o in r['outcomes']),'force_failures':sum(o['excessive_impact'] for o in r['outcomes']),'force_peak_quantiles_N':dict(zip(['p50','p90','p95','p99','max'],map(float,np.quantile(peaks,[.5,.9,.95,.99,1])))),'max_descent_m_s':r['maximum_descent_m_s'],'hold_quantiles_s':list(map(float,np.quantile(z['hold'].max(0),[0,.5,1]))),'final_linear_rms_max':max(o['final_3s_linear_rms_m_s'] for o in r['outcomes'])}
 if 'lin_vel' in z:
  row['max_body_speed_m_s']=float(np.linalg.norm(z['lin_vel'],axis=-1).max())
  first=(z['support']>.05).argmax(0);row['post_first_contact_root_descent_quantiles_m_s']=list(map(float,np.quantile(-z['lin_vel'][first,np.arange(len(first)),2],[.5,.9,1])))
  before=np.maximum(first-1,0);row['last_pre_contact_root_descent_quantiles_m_s']=list(map(float,np.quantile(-z['lin_vel'][before,np.arange(len(first)),2],[.5,.9,1])))
 if 'sitting_body_velocity' in z:
  row['pelvis_thigh_peak_speed_m_s']=float(np.linalg.norm(z['sitting_body_velocity'],axis=-1).max())
 if 'estimated_contact' in z:
  load=np.concatenate((np.zeros_like(z['support'][:1]),z['support'][:-1]));truth=load>.05;pred=z['estimated_contact'][...,0]>.65
  row['contact_false_positive_fraction']=float((pred&~truth).sum()/max(1,(~truth).sum()));row['contact_false_negative_fraction']=float((~pred&truth).sum()/max(1,truth.sum()));row['contact_load_mae']=float(abs(z['estimated_contact'][...,1]-np.clip(load,0,1)).mean())
  tp=int((pred&truth).sum());fp=int((pred&~truth).sum());fn=int((~pred&truth).sum())
  row['contact_precision']=tp/max(1,tp+fp);row['contact_recall']=tp/max(1,tp+fn)
  row['false_contact_episodes']=int((pred&~truth).any(0).sum())
  first_truth=truth.argmax(0);far_before=np.arange(len(truth))[:,None]<first_truth[None,:]-10
  row['false_contact_episodes_more_than_200ms_early']=int((pred&far_before).any(0).sum())
  delays=[]
  for i in range(len(peaks)):
   if truth[:,i].any() and pred[:,i].any():delays.append((pred[:,i].argmax()-truth[:,i].argmax())*.02)
  row['first_detection_delay_quantiles_s']=list(map(float,np.quantile(delays,[0,.5,1]))) if delays else []
 rows.append(row);axes[0].plot(np.sort(peaks),label=root.name);axes[1].plot(np.arange(len(z['support']))*.02,np.median(z['seat_force_peak'],axis=1),label=root.name)
axes[0].axhline(632.527832,color='r',ls='--');axes[0].set_ylabel('Per-attempt peak force (N)');axes[0].set_xlabel('Sorted attempts');axes[1].set_ylabel('Median seat force (N)');axes[1].set_xlabel('Time (s)');axes[0].legend(fontsize=7);fig.tight_layout();fig.savefig(out/'forces.png',dpi=150);plt.close(fig)
(out/'metrics.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
