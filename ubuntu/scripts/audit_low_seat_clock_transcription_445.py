"""Independent saved-array algebra audit of444, not full-path admission.
No FK, IK, AD, derivative fixture, finite-work audit or optimizer is repeated.
"""
from pathlib import Path
from fractions import Fraction
import json,hashlib,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');I=D/'clock_transcription_preparation_444';O=D/'clock_transcription_preparation_audit_445'
read=lambda p:json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def main():
 assert read(D/'clock_transcription_preparation_exit_444.json')['actual_exit']==0
 for p,h in read(I/'SHA256.json').items():assert sha(I/p)==h,p
 P=read(I/'preparation.json');assert P['passed'] and not P['optimizer_launch_allowed']
 O.mkdir(exist_ok=False);pr=dict(np.load(I/'primal_replay.npz'));df=dict(np.load(I/'derivative_fixtures.npz'));g=dict(np.load(I/'mesh.npz'));J=np.load(I/'local_blocks.npz')['J'];src=dict(np.load(D/'contact_lift_440/states.npz'));records=read(D/'contact_lift_440/records.json')
 X=pr['X'];Y=pr['Y'];assert X.shape==(1467,24) and Y.shape==(1467,51) and J.shape==(1467,51,24)
 errors={}
 for name,a,b,bound in [('command',Y[:,15:31],pr['reference_command'],2e-10),('torque',Y[:,31:47],pr['reference_torque'],1e-8),('com',Y[:,48:51],pr['reference_com'],2e-12),('gap',Y[:,47],pr['reference_gap'],2e-12),('moment',Y[:,12:14],pr['reference_equilibrium'][:,1:],2e-12),('normal',Y[:,14],pr['reference_equilibrium'][:,0],2e-12)]:
  e=float(abs(a-b).max());assert e<bound,(name,e);errors[name]=e
 # Rebuild exact geometric/force Jacobian identities independently, using
 # fixed canonical world points and direct rootxy variables, not oracle code.
 errors['force_moment_derivative']=float(abs(J[:,12:14,16:24]-src['points'][:,:,:2].transpose(0,2,1)).max())
 errors['normal_derivative']=float(abs(J[:,14,16:24]-1).max())
 errors['force_independent_geometry']=float(abs(J[:,:12,16:24]).max())
 errors['force_independent_gap_com']=float(abs(J[:,47:51,16:24]).max())
 errors['root_translation_com']=float(abs(J[:,48:51,:2]-np.eye(3)[:,:2]).max())
 errors['root_translation_moment']=float(abs(J[:,12:14,:2]+np.eye(2)).max())
 kp=np.array([200,150,150,200,160,160]*2+[40]*4)
 errors['force_command_chain']=float(abs(J[:,15:31,16:24]-J[:,31:47,16:24]/kp[None,:,None]).max())
 for k in ['force_moment_derivative','normal_derivative','force_independent_geometry','force_independent_gap_com','root_translation_com','root_translation_moment','force_command_chain']:assert errors[k]<2e-12,(k,errors[k])
 errors['fixture_FD_replay']=float(abs(df['J']-df['FD']).max());errors['fixture_directional_replay']=float(abs(np.einsum('nrc,c->nr',df['J'],df['direction'])-df['FD_direction']).max())
 assert max(errors['fixture_FD_replay'],errors['fixture_directional_replay'])<2e-6
 assert np.array_equal(J[df['indices']],df['J'])
 # Independent rational identity reconstruction. All1467 labels retained.
 times=[Fraction(int(r['exact_time'][0]),int(r['exact_time'][1])) for r in records];stored=[Fraction(int(a),int(b)) for a,b in read(I/'exact_times.json')];assert times==stored
 unique=sorted(set(times));lookup={t:k for k,t in enumerate(unique)};assert np.array_equal(g['label_to_time'],[lookup[t] for t in times])
 hover=times[1450];release=times[1451];keys={};canonical=[];ls=[]
 for i,t in enumerate(times):
  key=hover if hover<=t<=release else t
  if key not in keys:keys[key]=len(keys);canonical.append(1450 if key==hover else i)
  ls.append(keys[key])
 ls=np.array(ls);canonical=np.array(canonical);K=len(canonical)
 assert np.array_equal(canonical,g['canonical']) and np.array_equal(ls,g['label_to_state'])
 assert np.array_equal(g['reference'],X[canonical]);assert np.array_equal(g['canonical_Y'],Y[canonical]);assert np.array_equal(g['canonical_J'],J[canonical])
 rep=[next(i for i,t0 in enumerate(times) if t0==t) for t in unique];ordered=ls[rep]
 assert np.array_equal(rep,g['ordered_representative']) and np.array_equal(ordered,g['ordered_states'])
 keep=ordered[:-1]!=ordered[1:];assert np.array_equal(keep,g['edge_kept'])
 A=sparse.load_npz(I/'monotone_incidence.npz');R=sparse.load_npz(I/'clock_rate_incidence.npz');T=sparse.load_npz(I/'clock_target_incidence.npz')
 v=np.cos(np.arange(K)*.173);vv=np.sin(np.arange(K)*.37)
 direct=np.diff(v[ordered])[keep];rate=np.diff(v[ls[:676]]);target=10*v[ls[1:676]]-9*v[ls[:675]]
 errors['monotone_graph']=float(abs(A@v-direct).max());errors['clock_rate_graph']=float(abs(R@v-rate).max());errors['clock_target_graph']=float(abs(T@v-target).max())
 for n in ['monotone_graph','clock_rate_graph','clock_target_graph']:assert errors[n]<2e-14,(n,errors[n])
 dual=np.cos(np.arange(A.shape[0])*.21);errors['sparse_adjoint_dot']=float(abs(dual@(A@vv)-vv@(A.T@dual)));assert errors['sparse_adjoint_dot']<2e-12
 assert times[93]==Fraction(93,50) and times[287]==Fraction(287,50) and hover==Fraction(532,50) and release==Fraction(557,50) and times[1351+105]==Fraction(658,50) and times[675]==Fraction(27,2)
 assert np.all(g['inactive']==(src['weights'][canonical]==0))
 assert np.array_equal(g['fixed'],[times[i]==0 or times[i]>=Fraction(658,50) for i in canonical])
 alias_error=float(abs(X[canonical][ls]-X).max());assert alias_error==P['alias_max_coordinate_difference']
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,passed=True,arithmetic_only=True,errors=errors,all_labeled_records=1467,exact_distinct_times=len(unique),state_blocks=K,alias_max_coordinate_difference=alias_error,synthetic_hold_state=keys[hover],frame_drift_remains=True,physical_admission=False,complete_global_constraint_assembly=False,optimizer_launch_allowed=False,new_candidate=False,FK_calls=0,root_calls=0,finite_work_columns_repeated=0,actual_forward_filter_ticks=0,physics_ticks=0,policy_updates=0)
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
