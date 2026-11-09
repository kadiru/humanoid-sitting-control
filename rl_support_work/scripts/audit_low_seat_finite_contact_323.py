"""Independent raw-return and scalar-statistic replay of finite-contact322."""
from pathlib import Path
import json, hashlib, datetime, math
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01'); P=D/'contact_wrench_diagnostic_259'; O=D/'finite_contact_audit_323'
read=lambda p:json.loads(Path(p).read_text()); sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cross(a,b):
 a=np.asarray(a);b=np.asarray(b)
 return np.stack([a[...,1]*b[...,2]-a[...,2]*b[...,1],a[...,2]*b[...,0]-a[...,0]*b[...,2],a[...,0]*b[...,1]-a[...,1]*b[...,0]],axis=-1)
def rot(q):
 q=np.asarray(q,float);q=q/np.linalg.norm(q,axis=-1,keepdims=True);x,y,z,w=np.moveaxis(q,-1,0)
 return np.stack([1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w),2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w),2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)],axis=-1).reshape(*q.shape[:-1],3,3)
def main():
 result=read(D/'finite_contact_diagnosis_322/results.json');assert all(sha(p)==h for p,h in result['source_sha256'].items())
 O.mkdir(exist_ok=False);a=dict(np.load(D/'finite_contact_diagnosis_322/arrays.npz'));s=dict(np.load(P/'substep_states.npz'));R=rot(s['link_transforms'][...,3:])
 meta=read(P/'actuator_metadata.json');inertia=read(P/'static_inertial_model.json');cm=read(P/'contact_recorder_metadata.json');cr=np.load(P/'contact_returns.npz');cov=np.load(D/'ankle_impulse_296/arrays.npz')['coverage']
 rec={}
 for i,r in enumerate(cm['records']):
  if r['substep']>1600:rec[(r['substep'],r['side'],r['kind'])]=[cr[f'r{i}_a{j}'] for j in range(6 if r['kind']=='normal' else 4)]
 maxerr={};checks=0;npnt=fpnt=0
 def check(key,actual,expected,bound=1e-9):
  nonlocal checks
  actual=np.asarray(actual,float);expected=np.asarray(expected,float);assert actual.shape==expected.shape,(key,actual.shape,expected.shape)
  assert np.array_equal(np.isnan(actual),np.isnan(expected)),key
  valid=np.isfinite(expected);e=float(np.max(abs(actual[valid]-expected[valid]))) if valid.any() else 0.
  assert e<=bound,(key,e,bound);maxerr[key]=max(maxerr.get(key,0.),e);checks+=1
 for k in range(1,1001):
  for c in range(24):
   for side,name in enumerate(['left','right']):
    g=((k-1)*24+c)*2+side;sub=k+1600
    nraw=rec[(sub,name,'normal')];fraw=rec[(sub,name,'friction')]
    ns=int(nraw[5][c,0]);nn=int(nraw[4][c,0]);fs=int(fraw[3][c,0]);nf=int(fraw[2][c,0]);npnt+=nn;fpnt+=nf
    f=nraw[0][ns:ns+nn,0].astype(float);p=nraw[1][ns:ns+nn].astype(float);n=nraw[2][ns:ns+nn].astype(float);sep=nraw[3][ns:ns+nn,0].astype(float)
    fv=fraw[0][fs:fs+nf].astype(float);pp=fraw[1][fs:fs+nf].astype(float)
    b=meta['body_names'].index(name+'_ankle_roll_link');local=np.array(inertia['local_com'][inertia['body_names'].index(name+'_ankle_roll_link')]);pose=s['link_transforms'][k,c,b];rr=R[k,c,b];center=pose[:3]+rr@local;v=s['link_velocities'][k,c,b].astype(float)
    nv=v[:3]+cross(v[3:],p-center);vn=np.einsum('ij,ij->i',nv,n)
    def fd(points):
     loc=np.einsum('ji,nj->ni',rr,points-pose[:3]);old=np.einsum('ij,nj->ni',R[k-1,c,b],loc)+s['link_transforms'][k-1,c,b,:3]
     return (points-old)/.005
    vfd=np.einsum('ij,ij->i',fd(p),n);frv=v[:3]+cross(v[3:],pp-center);frfd=fd(pp)
    F=math.fsum(f);denom=F if F>0 else np.nan;positive=f>0
    C=np.concatenate([n,cross(p-center,n)/.1],axis=1)[positive];sv=np.linalg.svd(C,compute_uv=False) if len(C) else np.array([])
    check('normal_singular_values',a['normal_singular_values'][g],np.pad(sv,(0,6-len(sv))))
    exp=dict(substep=sub,case=c,side=side,observed_floor_only=int(cov[k-1,c]),normal_points=nn,loaded_points=int(positive.sum()),friction_points=nf,normal_rank=int((sv>1e-10).sum()),normal_force=F,normal_impulse=F*.005,
      positive_velocity_force=math.fsum(f[vn>0]),negative_velocity_force=math.fsum(f[vn<0]),zero_velocity_force=math.fsum(f[vn==0]),positive_gap_force=math.fsum(f[sep>0]),weighted_normal_velocity=math.fsum(f*vn)/denom,weighted_abs_normal_velocity=math.fsum(f*abs(vn))/denom,weighted_normal_fd=math.fsum(f*vfd)/denom,weighted_separation=math.fsum(f*sep)/denom,
      normal_force_endpoint_pairing=math.fsum(f*vn),normal_force_material_fd_pairing=math.fsum(f*vfd),friction_force_endpoint_pairing=math.fsum((fv*frv).ravel()),friction_force_material_fd_pairing=math.fsum((fv*frfd).ravel()),friction_resultant_norm=float(np.linalg.norm(np.array([math.fsum(fv[:,i]) for i in range(3)]))),friction_norm_sum=math.fsum(np.linalg.norm(fv,axis=1)),friction_norm_sum_over_normal=math.fsum(np.linalg.norm(fv,axis=1))/denom,friction_endpoint_positive_terms=int((np.einsum('ij,ij->i',fv,frv)>0).sum()),friction_endpoint_negative_terms=int((np.einsum('ij,ij->i',fv,frv)<0).sum()))
    for key,value in exp.items():check(key,a[key][g],value,0 if isinstance(value,int) else 1e-9)
 assert npnt==result['normal_point_records'] and fpnt==result['friction_point_records']
 statchecks=0
 for row in result['rows']:
  lo,hi=[round(t/.005) for t in row['window']];side=['left','right'].index(row['side']);m=(a['case']==row['case'])&(a['side']==side)&(a['substep']>=lo)&(a['substep']<=hi)
  assert m.sum()==row['samples'] and a['observed_floor_only'][m].sum()==row['observed_floor_only']
  for key,got in row['metrics'].items():
   vals=sorted(float(x) for x in a[key][m] if np.isfinite(x));N=len(vals)
   if not N:assert got is None;continue
   want=dict(count=N,min=vals[0],median=(vals[(N-1)//2]+vals[N//2])/2,max=vals[-1],rms=math.sqrt(math.fsum(x*x for x in vals)/N))
   for field,v in want.items():check('stat_'+field,got[field],v);statchecks+=1
  total=math.fsum(a['normal_force'][m])
  for field,source in [('force_weighted_positive_velocity_fraction','positive_velocity_force'),('force_weighted_negative_velocity_fraction','negative_velocity_force'),('force_weighted_positive_gap_fraction','positive_gap_force')]:
   if total:check(field,row[field],math.fsum(a[source][m])/total)
   else:assert row[field] is None
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,foot_intervals=48000,normal_point_records=npnt,friction_point_records=fpnt,case_windows=len(result['rows']),checks=checks,statistic_checks=statchecks,maximum_errors=maxerr,arithmetic_bound=1e-9,rank_threshold=1e-10,physical_admission=False,repair_complete=False,new_physics_ticks=0,method='Original normal/friction query slices, manual quaternion matrices and component cross products; independent scalar fsum/sorted statistics. All counts and source hashes exact. Numerical bounds are not physical/quiet threshold changes.',source_sha256={str(p):sha(p) for p in [Path(__file__).resolve(),D/'finite_contact_diagnosis_322/results.json',D/'finite_contact_diagnosis_322/arrays.npz',P/'contact_returns.npz',P/'substep_states.npz']})
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({'independent_audit.json':sha(O/'independent_audit.json')},indent=2));print(json.dumps(dict(passed=True,foot_intervals=48000,checks=checks,maxerror=max(maxerr.values()))),flush=True)
if __name__=='__main__':main()
