"""Offline replay exporter, reusing the preserved browser-only renderer template."""
import argparse,ast,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--preflight',action='store_true');a=p.parse_args();assert not a.output.exists()
 old=Path(__file__).with_name('render_two_surface_replay.py');tree=ast.parse(old.read_text());template=next(n.value.value for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='template' for t in n.targets))
 chains=[['pelvis','torso_link','head_link']]
 for side in ['left','right']:
  chains.append(['pelvis']+[side+s for s in ['_hip_pitch_link','_hip_roll_link','_hip_yaw_link','_knee_link','_ankle_pitch_link','_ankle_roll_link']]);chains.append(['torso_link']+[side+s for s in ['_shoulder_pitch_link','_shoulder_roll_link','_shoulder_yaw_link','_elbow_pitch_link','_elbow_roll_link','_palm_link']])
 keys=['fixed_all','reference_all','eval_run1']+([] if a.preflight else ['eval_run2']);payload={}
 for key in keys:
  chunks=[]
  for surface in ['sitting','leaning']:
   if a.preflight:
    suffix='02' if surface=='leaning' else '01';label={'fixed_all':f'{surface}_old_reference_01','reference_all':f'{surface}_reference_validation_{suffix}','eval_run1':f'{surface}_smoke_evaluation_{suffix}'}[key];folder=a.data/label
   else:
    seeds=read(a.data/surface/'protocol.json')['seeds'];label={'fixed_all':'old_reference','reference_all':'reference','eval_run1':f'policy_{seeds[0]}','eval_run2':f'policy_{seeds[1]}'}[key];folder=a.data/surface/label
   r=read(folder/'results.json');t=np.load(folder/'trace.npz');g=read(folder/'geometry.json');rows=[next(e for e in r['episodes'] if e['start_stage']==s) for s in range(7)];cases=[e['case'] for e in rows]
   names=list(dict.fromkeys(n for chain in chains for n in chain if n in r['body_names']));ids=[r['body_names'].index(n) for n in names];links=[[names.index(n) for n in ch if n in names] for ch in chains]
   xyz=(t['body'][:,:,:,:3]-np.array(r['origins'])[None,:,None,:])[:,cases][:,:,ids];xy=xyz[:,:,:,[0,2]]
   edges=[]
   for e in rows:
    geom=g[e['geometry_index']];c=np.array(geom['top_center']);u=Rotation.from_quat(geom['rotation_xyzw']).as_matrix()[:,0];edge=c[None]+np.array([-.5,.5])[:,None]*geom['depth']*u;edges.append(edge[:,[0,2]].round(4).tolist())
   chunks.append(dict(chains=links,xy=xy.round(4),edges=edges,episodes=rows,**{k:t[k][:,cases] for k in ['good','alive','support','traction','legal']},rms=np.sqrt(t['effort'][:,cases]/37).round(3)))
  item=dict(chains=chunks[0]['chains'],edges=sum([c['edges'] for c in chunks],[]),episodes=sum([c['episodes'] for c in chunks],[]))
  for k in ['xy','good','alive','support','traction','legal','rms']:item[k]=np.concatenate([c[k] for c in chunks],axis=1).tolist()
  payload[key]=item
 template=template.replace("['fixed_all','Fixed posture']","['fixed_all','Old reference']").replace("['reference_all','Reference only']","['reference_all','Corrected reference']")
 template=template.replace("return key.replace('eval_','Seed ').replace('_last_passing',' · last passed').replace('_last',' · last trained')", "return key==='eval_run1'?'"+('Two-update smoke' if a.preflight else 'PPO run 1 (separate policies)')+"':'PPO run 2 (separate policies)'")
 template=template.replace('Compare identical starting conditions.', 'Compare identical starting conditions; the first simulator placement is shown for each stage. '+('These are reference fixes and a two-update execution check, not final learned results.' if a.preflight else 'Each task has its own trained policy.'))
 a.output.write_text(template.replace('__DATA__',json.dumps(payload,separators=(',',':'),allow_nan=False)));print(a.output)
if __name__=='__main__':main()
