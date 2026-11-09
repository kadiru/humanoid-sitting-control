"""Independent saved-array allocation and new all-hard objective/row audit.
No import of428 construction; no robot model, root, projection or optimizer.
"""
from pathlib import Path
from fractions import Fraction
import math,json,hashlib,datetime,numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'lattice_preparation_audit_429'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);p=json.loads((D/'lattice_preparation_428/preparation.json').read_text());f=dict(np.load(D/'lattice_preparation_428/fixture.npz'));b=dict(np.load(D/'window_path_400/initial.npz'));old=dict(np.load(D/'monotone_preparation_410/fixture.npz'));s=dict(np.load(D/'budget_path_424/structural.npz'));proof=json.loads((D/'lattice_preparation_428/allocation_proof.json').read_text())
 assert p['passed'] and np.array_equal(f['reference'],s['z'][:894])
 parts=[]
 for k in range(114):
  rates=[];uppers=[];lowers=[]
  for j in range(16):
   dc=f['root_commands'][k+1,j]-f['root_commands'][k,j];ul=f['ulp'][j];up=b['command_hi'][j]-f['root_commands'][k+1,j]-32*ul;lo=f['root_commands'][k+1,j]-b['command_lo'][j]-32*ul
   assert up>0 and lo>0
   rates.append(abs(dc)/(b['rates'][j]-4*ul/.02));uppers.append(.18*max(dc,0)/up);lowers.append(.18*max(-dc,0)/lo)
  normal=(f['root_gaps'][k-1]-f['root_gaps'][k])/.006 if 99<=k<104 else 0.
  parts.append([.02,max(rates),max(uppers),max(lowers),normal])
 parts=np.array(parts);assert np.array_equal(parts,f['parts']);req=parts.max(1);assert np.array_equal(req,f['required'])
 groups=((0,16),(16,64),(64,99),(99,104),(104,114));phase=np.array([sum(req[a:c]) for a,c in groups]);assert np.array_equal(phase,f['phase_required'])
 lower=(88,189,241,101,11);upper=(98,199,251,106,21);budgets=tuple(map(int,f['budgets']));assert sum(budgets)==650 and all(a<=v<=b for a,v,b in zip(lower,budgets,upper))
 q=Fraction(proof['objective_numerator'],proof['objective_denominator']);ratios=[Fraction(float(r))*50/b for r,b in zip(phase,budgets)];assert max(ratios)==q
 # Independent threshold certificate proves the optimum WITHOUT another search.
 le=[];lt=[]
 for g,r in enumerate(phase):
  le.append(next((b for b in range(lower[g],upper[g]+1) if Fraction(float(r))*50/b<=q),None))
  lt.append(next((b for b in range(lower[g],upper[g]+1) if Fraction(float(r))*50/b<q),None))
 assert le==proof['minimum_ticks_at_optimum'] and lt==proof['minimum_ticks_strictly_below'];assert None not in le and sum(le)<=650;assert None in lt or sum(lt)>650
 # Replay every stored DP transition and tie choice; arithmetic only.
 state={0:(Fraction(0),())}
 for g,r in enumerate(phase):
  nxt={}
  for total in range(sum(lower[:g+1]),sum(upper[:g+1])+1):
   possibilities=[(max(state[total-b][0],Fraction(float(r))*50/b),state[total-b][1]+(b,)) for b in range(lower[g],upper[g]+1) if total-b in state]
   if possibilities:nxt[total]=min(possibilities)
  expected={int(k):(Fraction(v['value_numerator'],v['value_denominator']),tuple(v['budgets'])) for k,v in proof['dynamic_program_layers'][g].items()};assert nxt==expected;state=nxt
 assert state[650]==(q,budgets)
 eps=2.**-23;keep=f['keep'];scale=f['scale'];values=np.r_[old['iq'][keep]/scale[keep]-eps,old['v'][:786]-f['lo']-eps,f['hi']-old['v'][:786]-eps];J=np.concatenate([old['I'][keep,:894]/scale[keep,None],old['G'][:786,:894],-old['G'][:786,:894]],axis=0)
 assert np.array_equal(values,f['values']) and np.array_equal(J,f['jacobian']);assert J.shape==(28887,894)
 radius=np.array([1.]*785+[.1]*109);assert np.array_equal(radius,f['radius']);assert np.array_equal(f['lower'],old['z'][:894]-radius) and np.array_equal(f['upper'],old['z'][:894]+radius)
 val=.5*sum(((f['objective_test']-f['reference'])/radius)**2);grad=(f['objective_test']-f['reference'])/radius**2;oe=float(max(abs(grad-f['objective_gradient'])));assert abs(val-f['objective_value'])<1e-15 and oe<1e-14
 de=float(abs(J@f['direction']-f['independent_direction']).max());assert de<2e-12
 hashes=p['source_input_sha256'].copy()
 for path in [Path(__file__),D/'lattice_preparation_428/fixture.npz',D/'lattice_preparation_428/allocation_proof.json',D/'lattice_preparation_428/preparation.json']:hashes[str(path.resolve())]=sha(path)
 for path,h in hashes.items():assert sha(path)==h,path
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,budgets=list(budgets),optimal_saved_ratio=float(q),ratio_optimality_certificate=True,all_DP_transitions_replayed=True,required_duration_error=0.,all_hard_rows=28887,free_continuous_coordinates=894,feasibility_slack_variables=0,all_values_and_jacobian_exact=True,algebra_directional_error=de,objective_gradient_rebuild_error=oe,new_model_evaluations=0,optimizer_calls=0,integer_candidate_evaluated=False,physical_admission=False,source_sha256=hashes)
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({'independent_audit.json':sha(O/'independent_audit.json')},indent=2));print(json.dumps({k:v for k,v in out.items() if k!='source_sha256'},indent=2))
if __name__=='__main__':main()
