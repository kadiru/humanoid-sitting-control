"""Export a standalone side-view replay of saved, evaluated body transforms.

Uses offline simulator truth solely for visualization. No simulator or training.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


def main():
    p=argparse.ArgumentParser();p.add_argument('--study',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    results=json.loads((a.report/'results.json').read_text())
    target=a.report/'motion_replay.html';assert not target.exists()
    payload={}
    chains=[['pelvis','torso_link','head_link']]
    for side in ['left','right']:
        chains.append(['pelvis']+[side+s for s in ['_hip_pitch_link','_hip_roll_link','_hip_yaw_link','_knee_link','_ankle_pitch_link','_ankle_roll_link']])
        chains.append(['torso_link']+[side+s for s in ['_shoulder_pitch_link','_shoulder_roll_link','_shoulder_yaw_link','_elbow_pitch_link','_elbow_roll_link','_palm_link']])
    for label,r in results.items():
        t=np.load(a.study/label/'trace.npz');geometries=json.loads((a.study/label/'geometry.json').read_text())
        compact_names=list(dict.fromkeys(n for chain in chains for n in chain if n in r['body_names']))
        ids=[r['body_names'].index(n) for n in compact_names]
        legs=[[compact_names.index(n) for n in chain if n in compact_names] for chain in chains]
        xyz=t['body'][:,:,ids,:3]-np.array(r['origins'])[None,:,None,:]
        xy=xyz[:,:,:,[0,2]]
        edges=[]
        for g in geometries:
            center=np.array(g['top_center']);u=Rotation.from_quat(g['rotation_xyzw']).as_matrix()[:,0]
            edge=center[None]+np.array([-.5,.5])[:,None]*g['depth']*u
            edges.append(edge[:,[0,2]].round(4).tolist())
        payload[label]=dict(chains=legs,xy=xy.round(4).tolist(),edges=edges,good=t['good'].tolist(),alive=t['alive'].tolist(),support=t['support'].round(4).tolist(),rms=np.sqrt(t['effort']/37).round(3).tolist(),traction=t['traction'].tolist(),legal=t['legal'].tolist(),episodes=r['episodes'])
    template=r'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>Sitting and leaning — motion replay</title>
<style>body{font:16px/1.45 system-ui,sans-serif;color:#183338;background:#f8faf9;max-width:1280px;margin:30px auto;padding:0 20px}h1{font-size:28px;margin-bottom:5px}.controls{display:flex;gap:20px;align-items:center;flex-wrap:wrap;margin:24px 0}select,button{font:inherit;padding:8px;border:1px solid #b8c6c8;border-radius:6px;background:white}input{width:min(560px,65vw)}.panels{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:18px}.card{background:white;border:1px solid #d7e1df;border-radius:10px;padding:14px}.card h2{font-size:18px;margin:0 0 6px}canvas{width:100%;aspect-ratio:1/1.1}.metrics{font-size:14px}.outcome{font-weight:650;min-height:44px}.note{font-size:14px;color:#52656a}a{color:#267080}@media(max-width:800px){.panels{grid-template-columns:1fr}}</style>
<h1>Sitting and leaning: inspect the movement</h1><p>Compare identical starting conditions. Stage 0 starts standing; stage 6 starts already supported.</p>
<div class="controls"><label>Start <select id="condition"></select></label><label>PPO checkpoint <select id="policy"></select></label></div>
<div class="controls"><button id="play">Play</button><input id="time" type="range" min="0" max="199" step="1" value="0" aria-label="Replay time"><strong id="clock"></strong></div>
<div class="panels" id="panels"></div>
<p class="note">Side-view skeleton from saved simulation body transforms at 10 Hz; this is an offline visualization, not a camera image or a runtime controller input. Orange is the support surface; grey is the floor. Lateral motion is not visible. Contact forces are summarized by the normal support load, so an oblique support load is not the vertical fraction of body weight.</p>
<p class="note">Success requires the entire 20-second trial without a physical failure and continuous qualifying support for the final 5 seconds. “No stable support” can include a robot that stays standing. RMS torque is instantaneous RMS over all 37 joints, not energy.</p><p><a href="report.html">Back to verified report</a></p>
<script>const data=__DATA__;
const condition=document.querySelector('#condition'),policy=document.querySelector('#policy'),slider=document.querySelector('#time');
for(let i=0;i<14;i++){let o=document.createElement('option');o.value=i;o.textContent=(i<7?'40 cm seat':'45° leaning surface')+' · stage '+i%7+(i%7===0?' (standing)':i%7===6?' (supported)':'');condition.append(o)}
function name(key){return key.replace('eval_','Seed ').replace('_last_passing',' · last passed').replace('_last',' · last trained')}
for(const key of Object.keys(data).filter(x=>x.startsWith('eval_'))){let o=document.createElement('option');o.value=key;o.textContent=name(key);policy.append(o)}
let timer=null;
function draw(){
 document.querySelector('#clock').textContent=((+slider.value+1)/10).toFixed(1)+' s';
 const n=+condition.value,t=+slider.value;document.querySelector('#panels').replaceChildren();
 for(const [key,title] of [['fixed_all','Fixed posture'],['reference_all','Reference only'],[policy.value,name(policy.value)]]){
  const d=data[key],row=d.episodes[n];const card=document.createElement('section');card.className='card';
  const h=document.createElement('h2');h.textContent=title;card.append(h);
  const outcome=document.createElement('div');outcome.className='outcome';outcome.textContent='Full-trial result: '+(row.success?'stable support':Object.values(row.failures).some(Boolean)?'physical failure':'no stable support');outcome.style.color=row.success?'#24654d':Object.values(row.failures).some(Boolean)?'#ae442e':'#866119';card.append(outcome);
  const canvas=document.createElement('canvas');canvas.width=700;canvas.height=770;card.append(canvas);const c=canvas.getContext('2d');
  const X=x=>70+(x+.7)*470,Y=z=>715-z*470;c.fillStyle='#fff';c.fillRect(0,0,700,770);
  c.strokeStyle='#dce3e2';c.lineWidth=1;c.font='21px system-ui';c.fillStyle='#657674';
  for(let z=0;z<=1.4;z+=.2){c.beginPath();c.moveTo(35,Y(z));c.lineTo(665,Y(z));c.stroke();c.fillText(z.toFixed(1),1,Y(z)+7)}
  c.strokeStyle='#9a6e31';c.lineWidth=15;c.beginPath();c.moveTo(...[X(d.edges[n][0][0]),Y(d.edges[n][0][1])]);c.lineTo(X(d.edges[n][1][0]),Y(d.edges[n][1][1]));c.stroke();
  c.strokeStyle='#53686b';c.lineWidth=3;c.beginPath();c.moveTo(35,Y(0));c.lineTo(665,Y(0));c.stroke();
  const color=d.alive[t][n]?'#267b91':'#ad5139';c.strokeStyle=color;c.fillStyle=color;c.lineWidth=5;
  for(const chain of d.chains){c.beginPath();chain.forEach((id,j)=>{const xy=d.xy[t][n][id];if(j===0)c.moveTo(X(xy[0]),Y(xy[1]));else c.lineTo(X(xy[0]),Y(xy[1]))});c.stroke()}
  for(const xy of d.xy[t][n]){c.beginPath();c.arc(X(xy[0]),Y(xy[1]),4,0,2*Math.PI);c.fill()}
  c.fillStyle='#52656a';c.font='22px system-ui';c.fillText('Forward position →',210,758);
  const m=document.createElement('div');m.className='metrics';m.textContent='At this instant: '+(d.good[t][n]?'qualifying support':d.alive[t][n]?'support criteria not met':'physical failure latched')+' · RMS torque '+d.rms[t][n].toFixed(2)+' Nm · support load '+d.support[t][n].toFixed(2)+' BW · traction margin '+(d.traction[t][n]?'passes':'fails');card.append(m);
  document.querySelector('#panels').append(card);
 }
}
function stop(){if(timer)clearInterval(timer);timer=null;document.querySelector('#play').textContent='Play'}
document.querySelector('#play').onclick=()=>{if(timer){stop();return}if(+slider.value===199)slider.value=0;document.querySelector('#play').textContent='Pause';timer=setInterval(()=>{slider.value=Math.min(199,+slider.value+1);draw();if(+slider.value===199)stop()},100)};
slider.oninput=draw;condition.onchange=()=>{stop();slider.value=0;draw()};policy.onchange=draw;draw();
</script></html>'''
    target.write_text(template.replace('__DATA__',json.dumps(payload,separators=(',',':'),allow_nan=False)))
    print(target)


if __name__=='__main__':main()
