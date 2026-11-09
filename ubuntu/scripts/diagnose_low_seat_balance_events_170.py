"""Saved event labels and preceding measured geometry; no new physics."""
from pathlib import Path
import json,hashlib,datetime,ast
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');P=D/'balance_feedback_probe_164/candidate';O=D/'balance_event_diagnosis_170'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);r=read(P/'results.json');ev=dict(np.load(P/'every_tick.npz'));tr=dict(np.load(P/'trace.npz'));guards=read(P/'fixed_guard_records_132.json');rows=[]
 names=read(P/'contact_body_names.json');assert len(names)==ev['unexpected_body_normal'].shape[-1]
 tree=ast.parse(Path('scripts/run_supported_postures.py').read_text());contacts=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='CONTACTS' for x in n.targets));known={body for _,body,_ in contacts}
 for row in r['episodes']:
  c=row['case'];bad=np.flatnonzero(ev['wrong'][:,c]>10);bodies=[]
  for j,name in enumerate(names):
   if name in known:continue
   ids=np.flatnonzero(ev['unexpected_body_normal'][:,c,j]>10)
   if len(ids):bodies.append(dict(body=name,first_exceedance_s=float(ev['t'][ids[0]]),maximum_N=float(ev['unexpected_body_normal'][:,c,j].max())))
  first=float(ev['t'][bad[0]]) if len(bad) else None;near=None
  if first is not None:
   at=int(np.flatnonzero(tr['t']<first+1e-10)[-1]);t=float(tr['t'][at]);g=tr['geometry'][at,c];prior=[x for x in guards if x['case']==c and x['tick']*.02<=t+1e-10]
   near=dict(saved_pose_time_s=t,event_time_s=first,geometry_valid=bool(g[11]),geometry_age_s=float(g[12]),sampled_guard=prior[-1]['measured_guard'] if prior else None)
  rows.append(dict(case=c,first_forbidden_contact_s=first,forbidden_bodies_over10N=bodies,preceding_saved_measurement=near,physical_flags=row['failures'],illegal_ticks=row['illegal_ticks']))
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Original20ms>10N forbidden labels and individual forbidden-body force threshold exceedances. Contact channel order comes from saved contact_body_names.json, excluding the actual CONTACTS permitted-body set; it differs from robot body order. Historical156 used the wrong mapping and remains preserved as rejected. Geometry uses the latest saved0.1s sample at/before the event; no exact-event pose claim. All24 rows retained. Current geometry may be unavailable and is never replaced by truth or previous fits.',rows=rows,physics_ticks=0,source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),P/'results.json',P/'trace.npz',P/'every_tick.npz',P/'fixed_guard_records_132.json',P/'contact_body_names.json',Path('scripts/run_supported_postures.py').resolve()]})
 (O/'results.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2))
 print(json.dumps([x for x in rows if x['first_forbidden_contact_s'] is not None]),flush=True)
if __name__=='__main__':main()
