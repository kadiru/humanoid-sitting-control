"""Independent audit of new integer timing tangent preflight392, no FK or optimization.
Rebuild all FIVE budget derivative columns, softmax timing derivatives, allocation
identities and scalar directional sums over all20284 rows. Existing shape derivatives
are checked against saved finite differences, not claimed as newly analytic FK.
"""
from pathlib import Path
import numpy as np,json,hashlib,math,datetime
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');O=D/'integer_restoration_preparation_audit_393'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);r=json.loads((D/'integer_restoration_preparation_392/preparation.json').read_text());assert r['passed'];f=np.load(D/'integer_restoration_preparation_392/fixture.npz');base=np.load(D/'support_preparation_371/fixture.npz')
 for p,h in r['source_input_sha256'].items():assert sha(p)==h,p
 B=np.zeros((114,5));T=np.zeros((114,109));dt=[];off=0;groups=[(0,16),(16,64),(64,99),(99,104),(104,114)]
 for g,(a,b) in enumerate(groups):
  vals=list(map(float,f['z'][786+off:786+off+b-a-1]))+[0.];ex=[math.exp(x-max(vals)) for x in vals];w=[v/math.fsum(ex) for v in ex];extra=.02*(int(f['budgets'][g])-(b-a))
  dt.extend(.02+extra*v for v in w)
  for i in range(b-a):
   B[a+i,g]=.02*w[i]
   for j in range(b-a-1):T[a+i,off+j]=extra*w[i]*((1. if i==j else 0.)-w[j])
  off+=b-a-1
 terr=max(float(abs(B-f['B']).max()),float(abs(T-f['T']).max()),float(abs(np.array(dt)-f['x'][1026:]).max()));assert terr<1e-12
 C=np.vstack([base['basecmd'][0,base['moving']],f['Y'][:,14:30]]) if 'basecmd' in base.files else None
 # Only command endpoints enter these timing derivatives.
 cmd=f['Y'][:,14:30];rates=base['rates'];lo=base['command_lo'];hi=base['command_hi'];ulp=f['ulp'];J=np.zeros((20284,5))
 for i in range(114):
  for j in range(16):
   for start in [12212,14036]:J[start+i*16+j]=(float(rates[j])-4*float(ulp[j])/.02)*B[i]
   J[15860+i*16+j]=(float(hi[j])-float(cmd[i,j])-32*float(ulp[j]))*B[i]
   J[17684+i*16+j]=(float(cmd[i,j])-float(lo[j])-32*float(ulp[j]))*B[i]
 for j in range(5):J[19754+j]=.006*B[99+j]
 J[19759]=-np.array([math.fsum(B[:,g]) for g in range(5)])
 berr=float(abs(J-f['I'][:,-5:]).max());assert berr<1e-12;assert np.array_equal(f['E'][:,-5:],np.zeros((1,5)))
 dots=np.array([math.fsum(float(a)*float(b) for a,b in zip(row,f['direction'])) for row in np.vstack([f['E'],f['I']])]);doterr=float(abs(dots-f['analytic']).max());fderr=float(abs(dots-f['fd']).max());assert doterr<1e-12 and fderr<2e-6
 assert f['keep'].sum()==19891 and not f['keep'][19759];assert np.all(f['lower']<=f['upper']);assert f['lower'][898]>=0
 result=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope=__doc__,timing_error=terr,budget_derivative_error=berr,scalar_dot_error=doterr,directional_error=fderr,rows=20284,budget_columns=5,all_movement_budget_sum=650,unchanged_derivative_bound=2e-6,maximum_milp_runs=1,physical_admission=False,source_sha256={str(p.resolve()):sha(p) for p in [Path(__file__),D/'integer_restoration_preparation_392/preparation.json',D/'integer_restoration_preparation_392/fixture.npz']})
 (O/'independent_audit.json').write_text(json.dumps(result,indent=2));(O/'SHA256.json').write_text(json.dumps({p.name:sha(p) for p in O.iterdir() if p.is_file()},indent=2));print(json.dumps(result,indent=2))
if __name__=='__main__':main()
