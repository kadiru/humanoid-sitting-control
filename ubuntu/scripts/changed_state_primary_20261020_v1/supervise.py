"""One new-state primary; all workers share a45-second total deadline."""
import time
ORIGIN=time.monotonic()
import fcntl
import os
import resource
import secrets
import signal
import subprocess
import sys
from common import *

def kill_group(p):
    try:os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError:pass
def stop_signal(signum,frame):raise SystemExit(128+signum)
def run(stage,script,token):
    check(ORIGIN)
    env=dict(os.environ,CHANGED_PRIMARY_STAGE=stage,CHANGED_PRIMARY_TOKEN=token,PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONOPTIMIZE',None)
    begin=time.monotonic();log=(O/(stage+'.log')).open('x')
    p=subprocess.Popen([sys.executable,str(HERE/script),repr(ORIGIN)],stdout=log,
        stderr=subprocess.STDOUT,env=env,start_new_session=True)
    waited=0;killed=None;sent=None;error=None
    try:
        while True:
            waited,status,usage=os.wait4(p.pid,os.WNOHANG)
            if waited:break
            if time.monotonic()-ORIGIN>=CAP and killed is None:
                killed='shared_total_deadline';sent=time.monotonic();kill_group(p)
            time.sleep(.01)
    except BaseException as exc:
        error=exc;killed='supervisor_interrupted_or_failed';sent=time.monotonic()
        if waited!=p.pid:kill_group(p);_,status,usage=os.wait4(p.pid,0)
    end=time.monotonic();code=os.waitstatus_to_exitcode(status);p.returncode=code;log.close()
    done=bool(code==0 and killed is None and error is None and end-ORIGIN<CAP)
    result=dict(stage=stage,actual_exit=code,elapsed_s=end-begin,total_elapsed_s=end-ORIGIN,
        completed_within_deadline=done,hard_termination_reason=killed,
        termination_signal_sent_at=sent,overshoot_s=max(0.,end-ORIGIN-CAP),
        peak_rss_bytes=int(usage.ru_maxrss)*1024,worker_pid=p.pid,
        supervisor_error=None if error is None else repr(error))
    write(O/(stage+'_exit.json'),result)
    print(stage,code,round(end-ORIGIN,3),killed or 'completed',flush=True)
    if error is not None:raise error
    return result
def main():
    resource.setrlimit(resource.RLIMIT_AS,(LIMIT,LIMIT))
    assert sys.platform=='linux' and sys.flags.optimize==0
    for s in [signal.SIGTERM,signal.SIGHUP]:signal.signal(s,stop_signal)
    c=contract();check(ORIGIN)
    lock=open('/tmp/low_seat_changed_state_primary_20261020_v1.lock','a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert not workers({os.getpid(),os.getppid()})
    O.mkdir(exist_ok=False);token=secrets.token_hex(24)
    write(O/'admission.json',dict(origin=ORIGIN,token=token,contract_sha256=sha(HERE/'contract.json')))
    stages=[run('producer','producer.py',token)]
    if stages[0]['completed_within_deadline']:stages.append(run('audit','independent_return.py',token))
    ar=read(O/'audit.json') if (O/'audit.json').exists() else None
    elapsed=time.monotonic()-ORIGIN
    valid=bool(len(stages)==2 and all(s['completed_within_deadline'] for s in stages)
        and ar and ar['passed'] and elapsed<CAP)
    write(O/'result.json',dict(execution_and_audit_passed=valid,
        candidate_accepted=bool(valid and ar['candidate_accepted']),stages=stages,
        total_elapsed_s=elapsed,total_cap_s=CAP,address_space_bytes=LIMIT,
        no_retry=True,solves=int((O/'scipy_started.json').exists()),physical_trials=0,PPO_updates=0,holdout_trials=0,
        physical_admission=False,repair_complete=False))
    print('Changed-state primary audited:',valid,flush=True)

if __name__=='__main__':main()
