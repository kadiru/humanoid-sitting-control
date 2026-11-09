"""Prepare new finite lifted decoder and minimax LP algebra; no LP/model run."""
import datetime,json
from pathlib import Path
import numpy as np
from scipy import sparse
from low_seat_lifted_feasibility_490 import *
O=D/'lifted_feasibility_preparation_490'

def main():
 launch=read(D/'lifted_preparation_launch_490.json');assert launch['compile_actual_exit']==0
 for p,h in launch['source_input_sha256'].items():assert sha(p)==h,p
 O.mkdir(exist_ok=False);d=load();features=d['initial'];actual=d['seed']['Y'];b=blocks(d,features,actual,d['seed']['J']);s=intervals(d,features,d['drops'],1.)
 assert np.array_equal(features[:,:24],d['seed']['X'])
 for name,m in [('feature_basis',d['P']),('normal_drop_chain',d['N']),('all_rows_chain',b['A']),('consistency_chain',b['C']),('interval_matrix',s['H'])]:sparse.save_npz(O/(name+'.npz'),m)
 for mode,ceiling in [('primary',None),('witness',.5*b['phi'])]:
  w=assemble(d,b,s,ceiling);p=O/mode;p.mkdir()
  for key in ['matrix','equality']:sparse.save_npz(p/(key+'.npz'),w[key])
  np.savez_compressed(p/'arrays.npz',**{k:w[k] for k in ['rhs','cost','bounds','equality_rhs']});(p/'scope.json').write_text(json.dumps(dict(ceiling=ceiling,fixture_only=True,primary_result_not_solved=True)))
 np.savez_compressed(O/'reference.npz',features=features,actualY=actual,J=d['seed']['J'],source_features=d['source_actual'],r=b['r'],h=b['h'],e=b['e'],guard=b['guard'],feature_scale=d['feature_scale'],sigma=d['sigma'],control_feature_rows=d['control_feature_rows'],varying=d['varying'],active=d['active'],extra_rows=d['extra_rows'],lower=d['lo'],upper=d['hi'])
 meta={k:[pair(v) for v in s[k]] for k in ['lower','upper','widths','closed_lower','closed_upper']};meta.update(owners=s['owners'],drops=[pair(v) for v in d['drops']]);(O/'intervals.json').write_text(json.dumps(meta))
 # NEW combined full-coordinate arithmetic fixture. Old468's SAVED arithmetic
 # center is read, never recomputed and never used as a trajectory seed.
 center=features.copy();free=d['cm']['free'];pos={int(v):i for i,v in enumerate(free)}
 for i,v in enumerate(free):
  if int(v)%10<2:center.ravel()[d['control_feature_rows'][i]]=.5*(d['lo'][i]+d['hi'][i])
 for state in d['active']:
  ids=[pos[state*10+j] for j in range(2,10) if state*10+j in pos];slots=[int(free[i])%10+14 for i in ids]
  lo=list(map(fq,d['lo'][ids]));hi=list(map(fq,d['hi'][ids]));total=sum(map(fq,features[state,slots]),F(0));beta=(total-sum(lo))/(sum(hi)-sum(lo));assert 0<beta<1
  center[state,slots]=[float(l+beta*(u-l)) for l,u in zip(lo,hi)]
 cd=list(map(unpack,read(D/'exact_gap_chart_preparation_468/layout.json')['fixture_center']));center[d['normal_states'],71]=gaps(cd,d['dom'])[0];center=canonical(d,center)
 cs=intervals(d,center,cd,1.);zero=certificate(cs,np.zeros(d['P'].shape[1]));assert zero['passed']
 z,zz,zm=decode(d,center,cd,np.zeros(d['P'].shape[1]),cs);assert np.array_equal(z,center) and zz==cd
 directions=[];outputs=[];metadata=[];cols=d['gap_layout']['columns'];normal_room=min(min(v-l,u-v) for l,v,u in zip(d['dom']['lower'],cd,d['dom']['upper']))
 for pattern in range(3):
  t=np.empty(d['P'].shape[1]);n0=d['B'].shape[1]
  for j,c in enumerate(cols):
   sign=1. if (j+pattern)%3 else -1.
   if c['kind']=='normal_drop':t[j]=sign*float(normal_room/F(4*206))
   elif c['kind']=='gap':t[j]=sign*.001*d['feature_scale'][c['state'],71]
   elif c['kind']=='orientation':t[j]=sign*.001*.02
   else:
    state=c['state'];ids=[pos[state*10+v] for v in range(2,10) if state*10+v in pos];vals=center.ravel()[d['control_feature_rows'][ids]];room=float(np.min(np.minimum(vals-d['lo'][ids],d['hi'][ids]-vals)));t[j]=sign*room/32
  for i,row in enumerate(d['extra_rows']):t[n0+i]=(1. if (i+pattern)%2 else -1.)*.001*d['feature_scale'].ravel()[row]
  out,dd,mm=decode(d,center,cd,t,cs);directions.append(t);outputs.append(out);metadata.append(mm)
 np.savez_compressed(O/'finite_joint_fixtures.npz',origin=center,directions=np.asarray(directions),outputs=np.asarray(outputs));(O/'finite_fixture_metadata.json').write_text(json.dumps(dict(drop_origin=[pair(v) for v in cd],zero=zm,nonzero=metadata,fixture_is_not_candidate=True)))
 # An outward exact bound must reject without ever calling a finite encoder.
 mini=dict(H=sparse.csr_matrix([[1.]]),lower=[F(0)],upper=[F(1)],widths=[F(1)],owners=[['outward_fixture']]);bad=certificate(mini,np.array([-1e-13]));assert not bad['passed']
 gap_error=float(abs(features[:,71]-d['source_actual'][:,71]).max());normal_error=float(abs(features[:,38]-d['source_actual'][:,38]).max());soft=d['rows']['soft']&d['rows']['keep']
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,specification=SPEC,lifted_coordinates=d['P'].shape[1],consistency_rows=b['C'].shape[0],intervals=s['H'].shape[0],canonicalization=dict(X_exact=True,fixed_exact=bool(np.array_equal(features[d['mesh']['fixed']],d['source_actual'][d['mesh']['fixed']])),gap_change_m=gap_error,normal_change_BW=normal_error,auxiliary_foot_moment_change_mixed=float(abs(features[:,24:38]-d['source_actual'][:,24:38]).max()),source_actual_mechanics_unchanged=True),initial_phi=b['phi'],initial_auxiliary_soft_merit=max(0.,float(-b['r'][soft].min())),initial_consistency_scaled_max=float(abs(b['h']).max()),initial_actual_normal_max=float(abs(actual[:,14]).max()),all_original_equalities_retained=True,feature_scale_min=float(d['feature_scale'].ravel()[d['varying']].min()),sigma_min=float(d['sigma'][d['active']][:,OUTPUT].min()),new_joint_fixtures=3,new_direct_coordinates_covered=len(d['extra_rows']),zero_identity=True,outward_certificate=bad,gap_trust_encoding_error_bound_m=float(207*U),load_rounding_contract_unchanged=True,illustrative_witness_ceiling_not_solved=True,model_calls=0,root_calls=0,ad_calls=0,lp_calls=0,new_complete_path=False,physical_admission=False)
 (O/'preparation.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(out,indent=2),flush=True)
if __name__=='__main__':main()
