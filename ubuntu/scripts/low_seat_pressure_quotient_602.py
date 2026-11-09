"""Frozen necessary-condition test for one exact corner-pressure quotient."""
from pathlib import Path
import hashlib, json, time
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
R=Path('/home/kadir/humanoid-sitting')
SPEC=dict(
 scope='One NEW exact eligibility test for eliminating a local corner-pressure self-stress while leaving the full physical direction and all modeled outputs unchanged. NOT a force-only solve, LP, completed reduction, primal point or actual mechanics evaluation.',
 rationale='In exact ideal vertical-force mechanics four fixed corners have three wrench components. A possible pressure quotient would remove the one internal load mode per foot and represent its individual load bounds by existential interval constraints. That requires an EXACT null mode of the STORED coefficient system, not merely a continuum identity or a small residual. This test can disprove that premise before any projection is attempted.',
 selection='Fixed first nonfixed saved state1, original490 state block0, LEFT foot corners0,1,2,3. Saved466 pivot corner2 and original prior columns2,3,4 represent load0-load2, load1-load2, load3-load2. Require this exact metadata and basis incidence; no alternate state, foot, pivot, basis or conditioning search.',
 source='Immutable572 zero-face equalities, already bound by601 preservation and inherited573/559 proofs. Rows1378 and1379 are the first block moment-x/y equalities. Use their exact stored binary64 coefficients at original columns2,3,4. No model/J evaluation, reassembly, old proof replay, inverse, linear solve or594/586 graph evaluation.',
 equations='For the fixed2x3 moment submatrix H, calculate signed2x2 cofactors n=(H01*H12-H02*H11,H02*H10-H00*H12,H00*H11-H01*H10). If all n=0, stop inconclusive with no fallback. Otherwise H has rank2 and its real kernel is span(n). Inspect the other48 consistency rows of block0 in increasing original row order, skipping the two moment rows, and form exact d=r.n. Stop immediately at FIRST d!=0. Then det([H0;H1;r])=d!=0 proves rank3 on these three conserved-load coordinates, so NO nonzero local left-foot pressure change can keep all these saved equalities unchanged while all other original coordinates are held fixed.',
 negative_result='A nonzero exact defect rejects this proposed output-preserving local pressure quotient for the saved system. It does NOT prove LP/physical infeasibility, failure of the ideal continuum wrench identity, the cause of598 timeout, poor conditioning, or impossibility of a different coupled reduction. No threshold, rounding, equality waiver or approximate-null replacement.',
 zero_result='If all48 tests are exactzero, report only this necessary local premise passed. No full-state quotient, box projection, solver or witness is authorized by that result; additional complete frozen projection and independent audit would still be required. No second state or foot is tested in this run.',
 audit='Independent603 imports no602 producer or solver. Read original572 selected rows and original466 basis/layout. Verify all saved row coefficients bitwise, fixed association and first-failure prefix. Independently compute moment-cofactor values and the failing3x3 determinant through integer common-denominator Leibniz expansion, plus H.n=0 identities. No old coefficient graph/inversion/encoding/proof replay.',
 resources='ONE producer and ONE independent auditor, each240s from before scientific imports/global source hashes,40GiB address-space. Check hash chunks, operations and output serialization. Peak memory unmeasured. Completion timing includes main output manifest, excludes final tiny timing record/stdout/shutdown. No extension of45s robot LP; no new numerical algorithm or LP launch.',
 terminal='Unexpected source association, absent/nonfinite input, any error/deadline or first nonzero defect terminates this candidate route. No alternative block/method/parameter or repeated completed arithmetic. A mathematical obstruction is a completed negative result, not an execution failure.',
 physical='All original/inward rows, both consistency signs, physical gates, finite encoding, all64 successes and all8 unresolved failures remain unchanged. Main PPO and fresh holdout remain unstarted. LastSEALED601/report600/video307 unchanged. Preparation only, task unfinished.',
 limit_s=240, address_space_GiB=40, complete_reduction=False, numerical_algorithm=False, primal_witness=False, physical_admission=False)
def read(p):return json.loads(Path(p).read_text())
def check(start):
 if time.monotonic()-start>240:raise TimeoutError('240s pressure-quotient eligibility cap')
def sha(p,start=None):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):
   h.update(b)
   if start is not None:check(start)
 return h.hexdigest()
def workers(exclude):
 out=[]
 for p in Path('/proc').glob('[0-9]*/cmdline'):
  try:a=p.read_bytes().decode().split('\0')
  except (OSError,UnicodeError):continue
  if a and 'python' in a[0] and any((x.startswith('scripts/') or x.startswith(str(R/'scripts')+'/')) and Path(x).name!=exclude for x in a[1:]):out.append(a)
 return out
