"""Independent saved-sample arithmetic check for stance diagnosis219. No physics."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from humanoid_sitting.shared_frame import BodyModel
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'stance_audit_220'
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def median(x):
 a=sorted(float(v) for v in x);n=len(a);return (a[(n-1)//2]+a[n//2])/2

def main():
 O.mkdir(exist_ok=False);s=D/'stance_diagnosis_219/results.json';r=read(s);m=BodyModel(D/'cooked_model_50/model.json');names=read(D/'private_diagnostic_198/actuator_metadata.json')['joint_names'];assert m.names==names
 for p,h in r['source_sha256'].items():assert sha(p)==h,p
 nraw=njoint=0;maxerr=0.;cached={}
 for row in r['rows']:
  run=row['run'];c=row['case'];lo,hi=row['window']
  if run not in cached:
   P=D/run/'candidate';cached[run]=(dict(np.load(P/'trace.npz')),dict(np.load(P/'every_tick.npz')),read(P/'touchdown_raw_176.json') if (P/'touchdown_raw_176.json').exists() else [])
  tr,tk,rr=cached[run];ts=[k for k,t in enumerate(tr['t']) if lo-1e-9<=t<=hi+1e-9];ks=[k for k,t in enumerate(tk['t']) if lo-1e-9<=t<=hi+1e-9];raw=[x for x in rr if x['case']==c and lo-1e-9<=x['tick']*.02<=hi+1e-9]
  for j,z in enumerate(row['joints']):
   q=[float(tr['q'][k,c,j]) for k in ts];cmd=[float(tr['command'][k,c,j]) for k in ts];tau=[float(tr['torque'][k,c,j]) for k in ts]
   v=[median(q),max(q)-min(q),median([x-y for x,y in zip(cmd,q)]),median(tau)];saved=[z[k] for k in ['q_median_rad','q_range_rad','command_minus_q_median_rad','approximate_torque_median_Nm']];err=max(abs(np.array(v)-saved));maxerr=max(maxerr,float(err));assert err<1e-12;njoint+=1
   if raw:
    vel=[x['qd'][j] for x in raw];pos=[x['q'][j] for x in raw];assert abs(median(vel)-z['raw_qd_median_rad_s'])<1e-12;assert abs(median(abs(x) for x in vel)-z['raw_qd_abs_median_rad_s'])<1e-12
    assert sum(a*b<0 for a,b in zip(vel,vel[1:]))==z['raw_qd_sign_changes'];assert len(raw)==z['raw_qd_samples'];nraw+=1
   else:assert z['raw_qd_samples']==0 and z['raw_qd_median_rad_s'] is None
  for f,z in enumerate(row['feet']):
   loads=[float(tk['force'][k,c,5+f,2])/float(tk['weight'][k,c]) for k in ks];assert abs(median(loads)-z['normal_load_median_bw'])<1e-12
   assert abs(max(float(tk['slip'][k,c,5+f]) for k in ks)-z['max_loaded_slip_speed_m_s'])<1e-12
 assert len(r['rows'])==144 and njoint==5328 and nraw==592
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,case_windows=144,joint_records=njoint,raw_velocity_joint_records=nraw,max_arithmetic_error=maxerr,physics_ticks=0,optimizer_updates=0,physical_admission=False,source_sha256={str(s):sha(s),str(Path(__file__).resolve()):sha(__file__)},limitation='Validates saved-sample arithmetic and ordering, not cause or controller safety. Original physical and exact-preservation failures remain failed.')
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(out),flush=True)
if __name__=='__main__':main()
