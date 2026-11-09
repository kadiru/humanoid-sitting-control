"""One execution owner, fully timed, terminal on failure; no science in driver."""
import time
START=time.monotonic()
import json,os,signal,subprocess,sys,datetime
from pathlib import Path
S=Path(__file__).resolve().parent
with (S/'owner_claim.json').open('x') as f:json.dump(dict(pid=os.getpid(),utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),f)
assert not Path('/data/humanoid-sitting-perception/low_seat_contact_01/hard_row_attribution_20261023_v1').exists()
env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1');env.pop('PYTHONOPTIMIZE',None)
with (S/'owner.log').open('x') as log:
 p=subprocess.Popen([sys.executable,str(S/'supervise.py')],stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 try:p.wait(timeout=max(0,44.5-(time.monotonic()-START)));timeout=False
 except subprocess.TimeoutExpired:
  timeout=True;os.killpg(p.pid,signal.SIGTERM)
  try:p.wait(timeout=.2)
  except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
 elapsed=time.monotonic()-START
 record=dict(actual_exit=p.returncode,timeout=timeout,elapsed_through_child_exit_s=elapsed,total_cap_s=45,passed=p.returncode==0 and not timeout and elapsed<45,no_retry=True)
 with (S/'execution_exit.json').open('x') as f:json.dump(record,f,indent=2)
 print(json.dumps(record),flush=True)
 if not record['passed']:sys.exit(1)
