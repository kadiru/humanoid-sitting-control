"""Sequential producer and independent auditor, both under one stage clock."""
import time
import sys
ORIGIN=float(sys.argv[2])
STAGE=sys.argv[1]
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent/STAGE))
from common import *
import os
import resource
import secrets
import signal
import subprocess

def kill_group(p):
    try: os.killpg(p.pid,signal.SIGKILL)
    except ProcessLookupError: pass

def stop(signum,frame):
    raise SystemExit(128+signum)

def run(stage,script,token):
    deadline=PRODUCER_END if stage=='producer' else AUDIT_END
    check(ORIGIN);assert time.monotonic()-ORIGIN<deadline
    env=dict(os.environ,GAP_CYCLE_TOKEN=token,GAP_CYCLE_STAGE=stage,PYTHONDONTWRITEBYTECODE='1')
    env.pop('PYTHONOPTIMIZE',None)
    begin=time.monotonic();killed=None;waited=0
    with (O/(stage+'.log')).open('x') as log:
        p=subprocess.Popen([sys.executable,str(HERE/script),repr(ORIGIN)],stdout=log,
            stderr=subprocess.STDOUT,env=env,start_new_session=True)
        try:
            while True:
                waited,status,usage=os.wait4(p.pid,os.WNOHANG)
                if waited:break
                if time.monotonic()-ORIGIN>=deadline and killed is None:
                    killed='absolute_stage_deadline';kill_group(p)
                time.sleep(.01)
        except BaseException:
            if not waited:kill_group(p);os.wait4(p.pid,0)
            raise
    end=time.monotonic();code=os.waitstatus_to_exitcode(status);p.returncode=code
    record=dict(stage=stage,actual_exit=code,elapsed_s=end-begin,total_elapsed_s=end-ORIGIN,
        completed_within_deadline=bool(code==0 and killed is None and end-ORIGIN<deadline),
        stage_deadline_s=deadline,hard_termination_reason=killed,
        overshoot_s=max(0.,end-ORIGIN-deadline),peak_rss_bytes=int(usage.ru_maxrss)*1024,worker_pid=p.pid)
    write(O/(stage+'_exit.json'),record)
    print(stage,code,round(end-ORIGIN,3),killed or 'completed',flush=True)
    return record

def main():
    resource.setrlimit(resource.RLIMIT_AS,(LIMIT,LIMIT))
    assert sys.platform=='linux' and not sys.flags.optimize
    for s in (signal.SIGTERM,signal.SIGHUP):signal.signal(s,stop)
    contract();check(ORIGIN)
    token=secrets.token_hex(24)
    write(O/'admission.json',dict(origin=ORIGIN,token=token,contract_sha256=sha(HERE/'contract.json')))
    stages=[run('producer','producer.py',token)]
    # A killed/failed producer cannot create an accepted stage, even if partial files exist.
    if stages[0]['completed_within_deadline'] and time.monotonic()-ORIGIN<AUDIT_END:
        stages.append(run('audit','independent.py' if NAME=='C' else 'independent_return.py',token))
    audit=read(O/'audit.json') if (O/'audit.json').exists() else None
    valid=bool(len(stages)==2 and all(s['completed_within_deadline'] for s in stages)
        and audit and audit['passed'] and time.monotonic()-ORIGIN<CAP)
    result=dict(execution_and_audit_passed=valid,candidate_accepted=bool(valid and audit['candidate_accepted']),
        stages=stages,total_elapsed_s=time.monotonic()-ORIGIN,total_cap_s=CAP,address_space_bytes=LIMIT,
        audit_sha256=sha(O/'audit.json') if audit else None,no_retry=True,
        solves=int((O/'scipy_started.json').exists()),physical_trials=0,PPO_updates=0,holdout_trials=0,
        physical_admission=False,repair_complete=False)
    write(O/'result.json',result)
    # All file hashing is included before the owner seals its stage exit record.
    names=sorted(p.name for p in O.iterdir() if p.is_file() and p.name not in ('supervisor.log','manifest.json'))
    write(O/'manifest.json',{n:sha(O/n) for n in names})
    check(ORIGIN)
    print('stage_complete',NAME,valid,result['candidate_accepted'],flush=True)

if __name__=='__main__':main()
