"""One new complete nominal support/COM preparation. No optimization, physics or runtime.
The old exact COMxy prescription is replaced by an explicit original5mm support polygon;
foot poses/targets, forces/equilibrium, planning bounds and all original gates remain.
"""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from low_seat_support_problem_371 import make_problem
from low_seat_support_rows_371 import rows,transformed
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'support_preparation_371'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);p=make_problem();x=p['x0'].copy();eq,iq=p['evaluate'](x);E,I=p['evaluate'](x,True);free=p['free'];scale,keep,hard,groups,proof=rows(p)
 X=p['last']['X'].copy();Y,Q,T,C=p['calc'](X,True);old=p['seed_data'];oldq=np.array([e['q'] for e in old['dense']['path']])[1:];oldcmd=np.array([e['command'] for e in old['dense']['command_nodes']])[1:,p['moving']]
 reconstruction=dict(q=float(abs(Q-oldq).max()),command=float(abs(Y[:,14:30]-oldcmd).max()),com=float(abs(C[:,:2]-np.array(p['comxy'])).max()),kinematic=float(abs(Y[:103,:14]).max()))
 assert max(reconstruction.values())<1e-9,reconstruction
 rng=np.random.default_rng(371);direction=np.zeros(len(x));direction[free]=rng.normal(size=len(free));direction/=np.linalg.norm(direction);h=1e-5
 ep,ip=p['evaluate'](x+h*direction);xp=p['last']['X'].copy();yp=p['last']['Y'].copy()
 em,im=p['evaluate'](x-h*direction);xm=p['last']['X'].copy();ym=p['last']['Y'].copy()
 eqfd=(ep-em)/(2*h);iqfd=(ip-im)/(2*h);eqan=E@direction;iqan=I@direction;err=float(max(abs(eqfd-eqan).max(),abs(iqfd-iqan).max()));assert err<2e-6,err
 # A declared COM perturbation fixture exercises all103 new coordinates once,
 # independently from the directional pair. It is not a candidate/path search.
 test=x.copy();v=test[:1026].reshape(114,9);v[:103,2]=.001;v[:103,3]=-.001
 et,it=p['evaluate'](test);XT=p['last']['X'].copy();YT,QT,TT,CT=p['calc'](XT,True)
 removed=np.flatnonzero(~keep);assert np.array_equal(I[removed],np.zeros_like(I[removed]))
 assert np.array_equal(iq[removed],np.array([a['value'] for a in sorted(proof,key=lambda a:a['index'])]))
 slack=max(0.,float(max(-iq[keep&~hard]/scale[keep&~hard])))+1e-6
 ea,aug=transformed(eq,iq,scale,keep,hard,slack)
 np.savez_compressed(O/'fixture.npz',x=x,free=free,bounds=np.array(p['bounds']),eq=eq,ineq=iq,E=E[:,free],I=I[:,free],direction=direction,h=h,eq_fd=eqfd,ineq_fd=iqfd,eq_analytic=eqan,ineq_analytic=iqan,scale=scale,keep=keep,hard=hard,slack=slack,augmented=aug,null=p['null'],base_weights=p['wbase'],wcom=p['wcom'],points=p['PP'],side_mask=p['ssmask'],counts=p['counts'],effort=p['effort'],rates=p['rates'],command_lo=p['command_lo'],command_hi=p['command_hi'],lower=p['lower'],upper=p['upper'],support_h=p['support_h'],support_k=p['support_k'],support_com=p['support_com'],X=X,Y=Y,Q=Q,T=T,C=C,test_x=test,test_X=XT,test_Y=YT,test_Q=QT,test_T=TT,test_C=CT,plus_X=xp,plus_Y=yp,minus_X=xm,minus_Y=ym)
 paths=[Path(__file__).resolve(),Path('scripts/low_seat_support_problem_371.py').resolve(),Path('scripts/low_seat_support_rows_371.py').resolve(),Path('scripts/low_seat_feasibility_rows_365.py').resolve(),D/'multiknot_path_344/results.json',D/'multiknot_audit_345/independent_audit.json',D/'clearance_lift_path_330/results.json',D/'feasibility_path_367/final.npz',D/'feasibility_path_audit_368/independent_audit.json']
 hashes={str(a):sha(a) for a in paths};oldhashes=old['source_input_sha256']
 for a,hsh in oldhashes.items():assert sha(a)==hsh;hashes[a]=hsh
 r=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,reconstruction=reconstruction,raw_directional_error=err,unchanged_derivative_bound=2e-6,source_input_sha256=hashes,groups=groups,removed_structural_rows=proof,original_rows=19760,added_support_rows=len(p['support_h']),all_rows=len(iq),retained_rows=int(keep.sum()),hard_rows=int((keep&hard).sum()),soft_epigraph_rows=int((keep&~hard).sum()),free_path_variables=len(free),declared_path_variables=len(x),initial_slack=slack,initial_hard_minimum_scaled=float(min(iq[keep&hard]/scale[keep&hard])),initial_hover_equality=float(eq[0]),initial_time_s=float(.5+x[1026:].sum()),initial_is_not_claimed_feasible=True,new_COM_coordinate_units='.1m per dimensionless unit',contact_particular='old weights plus A pseudoinverse[:,1:]*COMdelta; inactive right foot exactlyzero',support='modeled stance sole convex hull inset original5mm; also mandatory actual FK finalgeometry',hard_constraints='contact bounds, effort, dependent planning bounds, monotone phases, all gap/descent, support polygon; hover equality exact; no epigraph slack',soft_constraints='command reserve, reserved rates, linear filter headroom,total13.5time; finaloriginalchecks unchanged',all_original_acceptance_criteria_retained=True,physical_admission=False,physics_ticks=0,nominal_optimizer_runs=0,policy_optimizer_updates=0)
 (O/'preparation.json').write_text(json.dumps(r,indent=2));(O/'SHA256.json').write_text(json.dumps({a.name:sha(a) for a in O.iterdir() if a.is_file()},indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['source_input_sha256','groups','removed_structural_rows']},indent=2),flush=True)
if __name__=='__main__':main()
