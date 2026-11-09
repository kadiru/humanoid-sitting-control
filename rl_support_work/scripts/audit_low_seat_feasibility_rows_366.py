"""Independent units, structural-constant and epigraph algebra audit of365; no FK/optimization/physics."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'feasibility_row_audit_366'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);prep=json.loads((D/'feasibility_preparation_365/preparation.json').read_text());a=np.load(D/'feasibility_preparation_365/fixture.npz');old=np.load(D/'reduced_path_v2_361/final.npz');initial=np.load(D/'reduced_path_v2_361/initial.npz')
 for p,h in prep['source_input_sha256'].items():assert sha(p)==h,p
 # Reconstruct the extraction exactly from the frozen361 source: only entry/output and unused helpers removed.
 src=Path('scripts/build_low_seat_reduced_path_v2_361.py').read_text();assert sha('scripts/build_low_seat_reduced_path_v2_361.py')=='9dbdb39072623305ed755bd6f72d6902685f23e0a060cdbe27ae35ae7200bd2b'
 original=src.split(' e0,i0=evaluate(x0);sourcefiles=')[0]
 extracted=Path('scripts/low_seat_reduced_problem_365.py').read_text()
 # Mathematical core from import torch through evaluate end is verbatim, including the inner solver, residuals and derivatives.
 core=original[original.index(' import torch,time'):]
 assert core==extracted[extracted.index(' import torch,time'):extracted.rindex(' return locals()')]
 assert np.array_equal(a['x'],old['x']) and np.array_equal(a['free'],initial['free'])
 assert max(abs(a['eq']-old['eq']).max(),abs(a['ineq']-old['ineq']).max())<1e-10
 scale=[];keep=[];removed=[];weight=np.asarray(initial['base_weights']);null=initial['null'];effort=np.array([300,300,300,300,20,20]*2+[300]*4,dtype=float);rate=np.asarray(initial['rates'])
 assert np.array_equal(effort,a['effort'])
 for i in range(114):
  for j in range(80):
   val=.03 if j<32 else effort[(j-32)%16] if j<64 else 1.
   scale.append(val);constant=False
   if j>=64:
    slot=(j-64)%8;constant=all(float(v)==0. for v in null[i,slot])
    if constant:
     value=float(weight[i,slot]) if j<72 else 1-float(weight[i,slot]);assert value>=0
     index=i*80+j;assert a['ineq'][index]==value;assert all(float(v)==0. for v in a['I'][index]);removed.append(index)
   keep.append(not constant)
 for i in range(103):
  for j in range(28):scale.append(.1 if j%14<2 else 1.);keep.append(True)
 for j in range(208):scale.append(1.);keep.append(True)
 for block in range(4):
  for i in range(114):
   for j in range(16):scale.append(float(rate[j])*.02 if block<2 else .0006);keep.append(True)
 for count,value in [(98,.012),(5,.012),(104,.012),(39,.012),(5,.00012),(1,13.5)]:
  for j in range(count):scale.append(value);keep.append(True)
 scale=np.array(scale);keep=np.array(keep);assert len(scale)==19760;assert np.array_equal(scale,a['scale']) and np.array_equal(keep,a['keep']);assert len(removed)==392
 reported=sorted(int(v['index']) for v in prep['removed_structural_rows']);assert reported==removed
 slack=float(a['slack']);expected=[]
 for i in range(19760):
  if keep[i]:expected.append(float(a['ineq'][i])/float(scale[i])+slack)
 for e in a['eq']:expected.append(float(e)+slack)
 for e in a['eq']:expected.append(-float(e)+slack)
 expected=np.array(expected);augerr=float(abs(expected-a['augmented']).max());assert augerr==0.;assert min(expected)>0
 # Audit every saved analytic directional dot product and the raw-unit derivative acceptance.
 v=a['direction'][a['free']];ea=np.array([sum(float(x)*float(y) for x,y in zip(row,v)) for row in a['E']]);ia=np.array([sum(float(x)*float(y) for x,y in zip(row,v)) for row in a['I']]);doterr=float(max(abs(ea-a['eq_analytic']).max(),abs(ia-a['ineq_analytic']).max()));assert doterr<1e-10
 derr=float(max(abs(a['eq_fd']-ea).max(),abs(a['ineq_fd']-ia).max()));assert derr<2e-6
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,full_original_rows=19760,structural_rows_removed_from_solver_only=len(removed),retained_rows=int(keep.sum()),epigraph_rows=len(expected),maximum_epigraph_replay_error=augerr,maximum_independent_scalar_dot_error=doterr,raw_derivative_fixture_error=derr,unchanged_derivative_bound=2e-6,all_original_rows_remain_final_checks=True,positive_slack_is_not_original_feasibility=True,model_core_verbatim=True,physical_admission=False,physics_ticks=0,policy_optimizer_updates=0,source_sha256={str(Path(__file__).resolve()):sha(__file__),str(D/'feasibility_preparation_365/preparation.json'):sha(D/'feasibility_preparation_365/preparation.json'),str(D/'feasibility_preparation_365/fixture.npz'):sha(D/'feasibility_preparation_365/fixture.npz')})
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
