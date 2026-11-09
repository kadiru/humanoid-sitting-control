"""Package the physical prerequisite separately from pending PPO outcomes."""
import argparse,ast,json,hashlib,shutil
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation
from run_progressive_seat_study import DATA,read,sha

def main():
 p=argparse.ArgumentParser();p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output;out.mkdir(exist_ok=False)
 names={'fixed':'fixed_01','fixed_x':'transfer_03','rolling':'rolling_fast_01'};data={};traces={};validation={}
 for key,name in names.items():
  path=DATA/name;data[key]=read(path/'results.json');traces[key]=np.load(path/'trace.npz');shutil.copytree(path,out/name)
  r=data[key];assert len(r['rows'])==18 and len(traces[key]['t'])==200
  if key in ['fixed','rolling']:
   assert r['strict_successes']==18
   assert all(x['good_ticks']==1000 and not any(x['failures'].values()) for x in r['rows'])
   assert all(all(x['slip_ok']) and not any(any(v) for v in x['failures'].values()) for x in read(path/'initial_settling.json'))
  validation[key]=dict(stable=r['successes'],strict=r['strict_successes'],physical=sum(any(x['failures'].values()) for x in r['rows']))
 for k in ['q','qd','command','root_pose','root_velocity']:
  assert np.array_equal(np.load(DATA/'fixed_01/startup.npz')[k],np.load(DATA/'rolling_fast_01/startup.npz')[k])
 geometry=read(DATA/'transfer_bank_01/geometry.json');metrics={}
 for pitch in [.1,.2]:
  ids=[i for i,g in enumerate(geometry) if g['near_pitch']==pitch];metrics[str(pitch)]={}
  for window,start in [('whole20s',0),('last10s',100),('last5s',150)]:
   t=traces['rolling']['torque'][start:,ids];b=traces['fixed']['torque'][start:,ids];am=float((t*t).sum(-1).mean());bm=float((b*b).sum(-1).mean())
   metrics[str(pitch)][window]=dict(rolling_Nm2=am,fixed_Nm2=bm,change_percent=100*(am/bm-1),rolling_RMS_Nm=float(np.sqrt((t*t).mean(-1)).mean()),seat_vertical_load=float(traces['rolling']['seat'][start:,ids].mean()),common_successes=9,sampling_seconds=.1)
 result=dict(passed=True,counts=validation,initial_states_exact=True,initial_settling_no_failure_or_slip=True,effort=metrics,scope='Near-seated physical reference validation, not PPO or standing success.100ms sampled torque metrics; all-tick stability/contact/friction/slip counts.')
 (out/'validation.json').write_text(json.dumps(result,indent=2))
 fig,axs=plt.subplots(2,2,figsize=(12,7.5),sharex=True)
 for col,pitch in enumerate([.1,.2]):
  ids=[i for i,g in enumerate(geometry) if g['near_pitch']==pitch]
  for key,label,color in [('fixed','Hold initial pose','#607d8b'),('fixed_x','Rotate with fixed pelvis x','#bb7452'),('rolling','Rotate about inferred seat contact','#247c73')]:
   t=traces[key];axs[0,col].plot(t['t'],(t['torque'][:,ids]**2).sum(-1).mean(1),label=label,color=color);axs[1,col].plot(t['t'],100*t['seat'][:,ids].mean(1),color=color)
  axs[0,col].set_title(f'Initial forward lean: {np.degrees(pitch):.1f}° · 9 surfaces');axs[0,col].set_ylabel('Sum of all37 joint torque² (Nm²)');axs[1,col].set_ylabel('Vertical weight on seat (%)');axs[1,col].set_xlabel('Seconds');axs[0,col].grid(alpha=.2);axs[1,col].grid(alpha=.2)
 axs[0,0].legend(fontsize=8);fig.suptitle('Near-seat weight transfer:18/18 clean transitions\nNo unwanted contacts, slipping or friction-margin violations at any active physics tick',fontsize=13);fig.tight_layout();fig.savefig(out/'validation.png',dpi=160);plt.close(fig)
 rows=''.join(f"<tr><td>{np.degrees(float(p)):.1f}°</td><td>9/9</td><td>{m['whole20s']['change_percent']:+.1f}%</td><td>{m['last5s']['change_percent']:+.1f}%</td><td>{100*m['last5s']['seat_vertical_load']:.1f}%</td></tr>" for p,m in metrics.items())
 (out/'findings.html').write_text(f'''<!doctype html><html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Near-seat transfer validation</title><style>body{{max-width:1080px;margin:40px auto;padding:0 24px;font:18px/1.5 system-ui;color:#20383d;background:#fafcfb}}h1{{line-height:1.2}}img{{width:100%}}table{{border-collapse:collapse;width:100%}}td,th{{padding:12px;border-bottom:1px solid #d1ded8;text-align:left}}.note{{color:#52676b}}.lead{{background:#e6f1ec;padding:20px;border-radius:10px}}</style><h1>Weight transfer without slipping</h1><p class="lead">The robot can move from a forward-leaning seated pose into a relaxed pose across all nine tested heights/slopes:18/18 trials, including two distinct starting poses. Every20-second trial passed all contact, slip and friction-margin checks.</p><p>The useful change was letting the pelvis rotate around its inferred contact with the seat. Forcing it to rotate at a fixed horizontal position trapped sideways forces between the feet and seat.</p><p><strong>This is a validated reference motion from near-seated starts. It is not a learned standing-to-sitting result.</strong> The two progressive PPO runs are a separate comparison; results will be reported after their fixed budgets and matched evaluations finish.</p><img src="validation.png" alt="Joint torque and weight transfer over20seconds"><table><tr><th>Starting lean</th><th>Clean trials</th><th>Whole20s torque²</th><th>Final5s torque²</th><th>Final seat weight</th></tr>{rows}</table><p class="note">Effort changes are compared with holding the identical starting pose, using all37 joints on common full-duration successes. Values use saved100ms samples. Torque is an effort proxy, not electrical energy. Both starting poses were already supported by the seat.</p><p>Geometry:38/40/42cm ×−5/0/+5°. The controller plans from depth, encoders, IMU and the body model with an explicit horizontal reference; simulator truth is used for reset construction and evaluation only. The harder bridge from the existing approach remains a learning challenge.</p><p><a href="validation.json">Validation details</a></p></html>''')
 shutil.copy2(__file__,out/'report_source.py');files=sorted(p for p in out.rglob('*') if p.is_file());(out/'SHA256.json').write_text(json.dumps({str(p.relative_to(out)):sha(p) for p in files},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
