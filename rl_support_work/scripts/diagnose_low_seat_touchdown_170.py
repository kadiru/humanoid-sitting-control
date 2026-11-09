"""Recorded touchdown slip and terminal-support timing, no controller changes."""
from pathlib import Path
import datetime,json,hashlib
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'balance_feedback_probe_164/candidate';O=D/'balance_touchdown_170'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);e=dict(np.load(P/'every_tick.npz'));results=read(P/'results.json')['episodes'];fb=read(P/'balance_feedback_161.json');out=[]
 for c in [0,1,6,7]:
  N=np.maximum(np.sum(e['force'][:,c]*e['normal'][:,c],-1),0)/e['weight'][:,c,None];legal=(N*e['rest_patch'][:,c]*e['permitted'][:,c]).sum(-1)
  ids=np.flatnonzero(~e['no_slip'][:,c]);around=np.flatnonzero((e['t']>=12.4)&(e['t']<=15.2));tail=np.flatnonzero(e['t']>=15.)
  rows=[]
  for k in around:
   r=next((x for x in fb if x['case']==c and x['tick']==round(e['t'][k]/.02)),None)
   rows.append(dict(t=float(e['t'][k]),normal_load_bw=N[k].tolist(),tangential_speed_m_s=e['slip'][k,c].tolist(),legal_seat_load_bw=float(legal[k]),no_slip=bool(e['no_slip'][k,c]),feedback=r['feedback'] if r else None))
  low=tail[legal[tail]<.05]
  out.append(dict(case=c,first_slip_s=float(e['t'][ids[0]]) if len(ids) else None,last_slip_s=float(e['t'][ids[-1]]) if len(ids) else None,slip_ticks=int(len(ids)),maximum_pelvis_slip_speed_during_flagged_ticks=float(e['slip'][ids,c,0].max()) if len(ids) else None,maximum_floor_contact_speed_during_flagged_ticks=float(e['slip'][ids,c,5:7].max()) if len(ids) else None,final_hold_s=results[c]['final_hold_s'],last_subthreshold_legal_seat_sample_after15s=float(e['t'][low[-1]]) if len(low) else None,series=rows))
 obj=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,rows=out,physics_ticks=0,optimizer_updates=0,physical_admission=False,limitations='Exact20ms saved slip/load labels. No new criterion. Event attribution uses saved fixed CONTACTS order: pelvis index0, feet5/6. Legal load follows saved normals/rest_patch/permitted; evaluation labels never enter runtime. No single-cause proof.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'every_tick.npz',P/'results.json',P/'balance_feedback_161.json']})
 (O/'results.json').write_text(json.dumps(obj,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps([{k:v for k,v in r.items() if k!='series'} for r in out]),flush=True)
if __name__=='__main__':main()
