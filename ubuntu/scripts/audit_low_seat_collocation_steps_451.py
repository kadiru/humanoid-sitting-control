"""Independent replay of saved450 LP witnesses, nonlinear rows and decisions.
No optimizer, root, model or derivative fixture is rerun here.
"""
from pathlib import Path
import json,datetime
import numpy as np
from scipy import sparse
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
def audit_steps(output):
 O=D/'collocation_path_450';V=D/'clock_row_preparation_v2_446';Q=D/'clock_row_preparation_446';row=dict(np.load(V/'row_arrays.npz'));M=sparse.load_npz(V/'all_rows_matrix.npz');E=sparse.load_npz(Q/'equality_matrix.npz');ea=dict(np.load(Q/'equality_arrays.npz'));coord=dict(np.load(Q/'coordinate_map.npz'));mesh=dict(np.load(D/'clock_transcription_preparation_444/mesh.npz'));initial=np.load(O/'initial.npz');res=json.loads((O/'results.json').read_text());history=[json.loads(x) for x in (O/'history.jsonl').read_text().splitlines()];spec=res['specification'];zref=coord['reference'].ravel()[coord['free']];lo=np.maximum(coord['lower'].ravel()[coord['free']],zref-.02);hi=np.minimum(coord['upper'].ravel()[coord['free']],zref+.02);hard=row['keep']&~row['soft'];soft=row['keep']&row['soft'];stateX=initial['X'].copy();stateY=initial['Y'].copy();maximum_replay=0.;linear_worst=0.;accepted=0;trials=0;lpcount=0;details=[]
 for outer,h in enumerate(history):
  assert h['predictor']==outer
  sr=(M@np.c_[stateX,stateY].ravel()+row['constant'])/row['scale'];merit=max(0.,float(-sr[soft].min()));assert abs(merit-h['start']['soft_merit'])<2e-10
  found=False;predicted=None
  for at in h['attempts']:
   k=at['linear_index'];assert k==lpcount;lpcount+=1;P=O/f'linear_{k:02d}';a=dict(np.load(P/'input.npz'));A=sparse.load_npz(P/'matrix.npz');F=sparse.load_npz(P/'equality.npz');meta=json.loads((P/'solver.json').read_text());entry=dict(linear=k,solver_status=meta['status'],candidate=meta['candidate'])
   if meta['candidate']:
    w=np.load(P/'witness.npy');v=max(0.,float((A@w-a['rhs']).max()),float(abs(F@w-a['equality_rhs']).max()),float((a['bounds'][:,0]-w).max()),float((w-a['bounds'][:,1]).max()));linear_worst=max(linear_worst,v);assert abs(v-meta['independent_linear_violation'])<2e-10;assert (v<=1e-7)==meta['linear_pass'];entry['independent_linear_violation']=v
    if at['correction']==0:predicted=merit-float(w[-1])
   if (P/'nonlinear_trial.npz').exists():
    trials+=1;t=np.load(P/'nonlinear_trial.npz');X=t['X'];Y=t['Y'];r=(M@np.c_[X,Y].ravel()+row['constant'])/row['scale'];e=E@np.c_[X,Y].ravel()+ea['constant'];z=X[:,coord['control']].ravel()[coord['free']]
    if (P/'nonlinear_rows.npz').exists():
     saved=np.load(P/'nonlinear_rows.npz');maximum_replay=max(maximum_replay,float(abs(r-saved['rows']).max()),float(abs(e-saved['equalities']).max()));actual=merit-max(0.,float(-r[soft].min()));ratio=actual/predicted;bounds=min(float((z-lo).min()),float((hi-z).min()));valid=bool(np.all(r[hard]>=0) and bounds>=0 and np.array_equal(X[mesh['fixed']],initial['X'][mesh['fixed']]) and np.all(X[:,16:24][mesh['inactive']]==0) and np.all(X[:,16:24]>=0) and np.max(abs(e[:-1]))<=1e-13 and abs(e[-1])<=1e-8)
     take=bool(valid and actual>0 and ratio>=.1);assert take==at['accepted'];entry.update(hard_pass=bool(np.all(r[hard]>=0)),maximum_total_normal=float(abs(e[:-1]).max()),hover_error=float(abs(e[-1])),accepted=take)
     if take:accepted+=1;found=True;stateX=X.copy();stateY=Y.copy()
    else:assert not at['root_pass'];entry['root_failed']=True
   details.append(entry)
  assert found==h['accepted']
 final=np.load(O/'final.npz');assert np.array_equal(final['X'],stateX) and np.array_equal(final['Y'],stateY);assert lpcount==res['linear_programs'] and accepted==res['accepted_steps'] and trials==res['nonlinear_trial_paths'];assert len(history)<=6 and lpcount<=18
 out=dict(passed=True,linear_programs=lpcount,linear_witness_worst_violation=linear_worst,nonlinear_trial_paths=trials,accepted_steps=accepted,maximum_saved_row_replay=maximum_replay,final_selection_exact=True,details=details,optimizer_calls=0,root_calls=0,model_calls=0,physical_admission=False);(output/'step_audit.json').write_text(json.dumps(out,indent=2));return out
