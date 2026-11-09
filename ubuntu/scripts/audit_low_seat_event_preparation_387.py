"""Independent event-duration, all675 clock rows, and derivative preflight audit.
Reads saved386v2 arrays; does not call its construction or redo FK/projection.
No optimizer, physics or candidate float32 schedule.
"""
from pathlib import Path
import json,math,hashlib,datetime,numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'event_clock_preparation_audit_387'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);r=json.loads((D/'event_clock_preparation_v2_386/preparation.json').read_text());assert r['passed'];f=np.load(D/'event_clock_preparation_v2_386/fixture.npz');old=np.load(D/'support_path_373/final.npz');base=np.load(D/'support_preparation_371/fixture.npz')
 for p,h in r['source_input_sha256'].items():assert sha(p)==h
 groups=[(0,16),(16,64),(64,99),(99,104),(104,114)];counts=json.loads((D/'discrete_path_379/results.json').read_text())['discrete']['segment_ticks'];budgets=[sum(counts[a:b]) for a,b in groups];assert budgets==list(f['budgets'])==[93,194,246,101,16]
 durations=[];T=np.zeros((114,109));off=0
 for (a,b),budget in zip(groups,budgets):
  values=list(map(float,f['z'][786+off:786+off+b-a-1]))+[0.];top=max(values);exps=[math.exp(x-top) for x in values];summed=math.fsum(exps);weights=[x/summed for x in exps];extra=.02*(budget-(b-a));durations.extend(.02+extra*w for w in weights)
  for i in range(b-a):
   for j in range(b-a-1):T[a+i,off+j]=extra*weights[i]*((1. if i==j else 0.)-weights[j])
  off+=b-a-1
 dt=np.array(durations);terr=max(float(abs(dt-f['x'][1026:]).max()),float(abs(T-f['T']).max()));assert terr<1e-12
 assert np.array_equal(f['x'][:1026],old['x'][:1026]);C=f['C'];assert np.array_equal(C[1:],f['Y'][:,14:30]);assert np.array_equal(f['X'],old['full_coordinates']) and np.array_equal(f['Y'],old['local_outputs'])
 samples=[C[0].tolist()];overlaps=[];event_ticks=[]
 for (a,b),budget in zip(groups,budgets):
  starts=[math.fsum(durations[a:i]) for i in range(a,b+1)]
  for tick in range(1,budget+1):
   t=tick*.02;w=np.zeros(114)
   for j in range(a,b):w[j]=max(0.,min(t,starts[j-a+1])-max((tick-1)*.02,starts[j-a]))
   overlaps.append(w)
   if tick==budget:samples.append(C[b].tolist())
   else:
    j=next(j for j in range(a,b) if t<starts[j-a+1]);alpha=(t-starts[j-a])/dt[j];samples.append([float(C[j,k])+(float(C[j+1,k])-float(C[j,k]))*alpha for k in range(16)])
  event_ticks.append(len(samples)-1)
  if b==99:
   for _ in range(25):samples.append(C[b].tolist());overlaps.append(np.zeros(114))
 S=np.array(samples);W=np.array(overlaps);assert S.shape==(676,16) and W.shape==(675,114);serr=float(abs(S-f['clock']).max());assert serr<1e-10
 delta=np.diff(S,axis=0);v=np.diff(C,axis=0)/dt[:,None];overlap_error=float(abs(W@v-delta).max());assert overlap_error<1e-10;assert np.all(W>=0) and abs(W.sum(1)[W.sum(1)>0]-.02).max()<1e-12
 rates=base['rates'];lo=base['command_lo'];hi=base['command_hi'];u=np.array([[10*float(S[k+1,j])-9*float(S[k,j]) for j in range(16)] for k in range(675)])
 raw=np.c_[.02*rates-delta,.02*rates+delta,hi-u,u-lo];eps=2.**-23;mag=np.maximum(abs(base['desired_lo']),abs(base['desired_hi'])) if 'desired_lo' in base.files else None
 ulp=f['ulp'];clock_tight=np.r_[4*ulp,4*ulp,32*ulp,32*ulp];full=raw-clock_tight;cerr=float(max(abs(raw-f['clock_raw']).max(),abs(full-f['clock_full']).max()));assert cerr<1e-10
 original=old['ineq'].copy();dc=np.diff(C,axis=0);original[12212:14036]=(.02*0+dt[:,None]*rates-dc).ravel();original[14036:15860]=(dt[:,None]*rates+dc).ravel();original[15860:17684]=((hi-C[1:])*dt[:,None]-.18*dc).ravel();original[17684:19508]=((C[1:]-lo)*dt[:,None]+.18*dc).ravel();g=np.r_[json.loads((D/'support_path_373/results.json').read_text())['dense']['path'][0]['pelvis_gap_m'],f['Y'][:,54]];original[19754:19759]=.006*dt[99:104]-(g[99:104]-g[100:105]);original[19759]=13.-dt.sum();oerr=float(abs(original-f['raw_ineq']).max());assert oerr<1e-10
 tight=np.zeros(20284);keep=f['keep'][:20284]
 for i in range(114):
  for j in range(8):
   for offset in [64,72]:
    if keep[i*80+offset+j]:tight[i*80+offset+j]=8*eps
 tight[12004:12212]=8*eps;tight[19760:]=8*eps
 for start in [12212,14036]:tight[start:start+1824]=(dt[:,None]*4*ulp/.02).ravel()
 for start in [15860,17684]:tight[start:start+1824]=(dt[:,None]*32*ulp).ravel()
 assert abs(tight-f['tighten']).max()<1e-14
 iq=np.r_[original-tight,full[:,32:].ravel()];assert abs(iq-f['ineq']).max()<1e-10 and len(iq)==41884
 dot=np.r_[[math.fsum(float(x)*float(y) for x,y in zip(row,f['direction'])) for row in f['E']],[math.fsum(float(x)*float(y) for x,y in zip(row,f['direction'])) for row in f['I']]];derror=float(abs(dot-f['analytic']).max());assert derror<1e-12;assert abs(dot-f['fd']).max()<2e-6
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,event_budgets=budgets,mandatory_event_tick_indices_including_quiet=event_ticks,timing_reconstruction_error=terr,clock_reconstruction_error=serr,overlap_increment_error=overlap_error,original_row_error=oerr,clock_row_error=cerr,scalar_dot_error=derror,directional_error=float(abs(dot-f['fd']).max()),all_clock_inequalities=43200,full_target_rows_in_optimizer=21600,all_original_support_rows=20284,optimizer_inequalities=41884,continuous_rate_implies_ideal_clock_rate_by_nonnegative_overlap=True,constant_holds=25,actual_float32_candidate_schedule=False,intermediate_geometry_checked=False,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),D/'event_clock_preparation_v2_386/preparation.json',D/'event_clock_preparation_v2_386/fixture.npz',D/'support_path_373/final.npz',D/'discrete_path_379/results.json']})
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
