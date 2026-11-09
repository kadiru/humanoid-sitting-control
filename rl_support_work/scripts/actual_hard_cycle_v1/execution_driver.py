"""Exclusive cycle owner. No retries, science without approval, or unchecked downstream."""
import time
START=time.monotonic()
import os
import sys
import signal
import resource
import fcntl
import shutil
import subprocess
from policy import authorize,advance,ROOT,STAGES,LIMIT,HELPER_LIMIT,sha,read
from stage_common import CYCLE,D,write,workers

def stop(signum,frame):
    raise SystemExit(128+signum)

def run(stage,origin):
    cap=STAGES[stage][0];folder=CYCLE/stage
    folder.mkdir(exist_ok=False)
    env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
    env.pop('PYTHONOPTIMIZE',None)
    interrupted=False
    with (folder/'supervisor.log').open('x') as log:
        p=subprocess.Popen([sys.executable,str(ROOT/'supervise.py'),stage,repr(origin)],
            stdout=log,stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            p.wait(timeout=max(0.,cap-.5-(time.monotonic()-origin)))
        except BaseException as exc:
            interrupted=True
            try:os.killpg(p.pid,signal.SIGTERM)
            except ProcessLookupError:pass
            try:p.wait(timeout=.2)
            except subprocess.TimeoutExpired:
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                p.wait()
            if not isinstance(exc,subprocess.TimeoutExpired):raise
    elapsed=time.monotonic()-origin
    record=dict(actual_exit=p.returncode,timeout=interrupted,total_cap_s=cap,
        elapsed_through_child_exit_s=elapsed,passed=bool(p.returncode==0 and not interrupted and elapsed<cap),
        no_retry=True,manifest_sha256=sha(folder/'manifest.json') if (folder/'manifest.json').exists() else None)
    write(folder/'execution_exit.json',record)
    # Downstream decisions charge parsing and exit-file finalization to this stage too.
    result=read(folder/'result.json') if (folder/'result.json').exists() else dict(execution_and_audit_passed=False,candidate_accepted=False)
    next_stage=advance(stage,result,record)
    elapsed=time.monotonic()-origin
    if elapsed>=cap:
        record.update(passed=False,total_finalized_seconds=elapsed)
        write(folder/'execution_exit.json',record);next_stage=None
    else:
        record['total_finalized_seconds']=elapsed
        write(folder/'execution_exit.json',record)
    return next_stage,record,result

def main():
    assert sys.platform=='linux' and not sys.flags.optimize
    resource.setrlimit(resource.RLIMIT_AS,(HELPER_LIMIT,LIMIT))
    for s in (signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    authorize()  # Missing/false approval prevents owner claim, data reads and all stage starts.
    lock=open('/tmp/low_seat_scientific_execution.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not workers({os.getpid(),os.getppid()})
    assert shutil.disk_usage(D).free>=4*1024**3
    CYCLE.mkdir(exist_ok=False)
    with (ROOT/'owner_claim.json').open('x') as f:
        import json
        json.dump(dict(pid=os.getpid(),start=START,freeze_sha256=sha(ROOT/'freeze.json')),f)
    records=[];stage='A';origin=START
    while stage:
        next_stage,record,result=run(stage,origin)
        records.append(dict(stage=stage,exit=record,result=result))
        # Completion bookkeeping is charged before starting a new independent stage clock.
        write(CYCLE/'cycle_result.json',dict(stages=records,closed=next_stage is None,
            research_success=bool(stage=='C' and record['passed'] and result['candidate_accepted']),
            physical_success=False,repair_complete=False,simulation=0,PPO=0,holdout=0,no_retry=True))
        if time.monotonic()-origin>=STAGES[stage][0]:
            write(CYCLE/'late_finalization_failure.json',dict(stage=stage,passed=False));break
        stage=next_stage
        origin=time.monotonic()

if __name__=='__main__':main()
