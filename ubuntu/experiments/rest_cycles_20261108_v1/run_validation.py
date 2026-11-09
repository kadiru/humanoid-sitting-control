"""Finite CPU owner. No simulation imports, GPU allocation or retries."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import resource
import subprocess
import sys
import time

HERE=Path(__file__).resolve().parent


def main():
    out=Path(sys.argv[1]);assert not out.exists(), 'OUTPUT_EXISTS_NO_RETRY'
    for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','NUMEXPR_NUM_THREADS']:
        os.environ[k]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    available=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{available[0]})
    resource.setrlimit(resource.RLIMIT_AS,(40*1024**3,40*1024**3))
    resource.setrlimit(resource.RLIMIT_CPU,(43,44))
    before=resource.getrusage(resource.RUSAGE_CHILDREN);started=time.monotonic()
    commands=[['test_core.py'],['contact_observer_check.py',str(out)],['audit_observer.py',str(out)]]
    receipt=dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 wall_cap_seconds=45,address_space_GiB=40,CPU_threads=1,CPU_affinity=[available[0]],
                 commands=[],scientific_attempts=1,retries=0,new_physics_steps=0,PPO_updates=0)
    try:
        freeze=json.loads((HERE/'freeze.json').read_text())
        for p,h in freeze['source_sha256'].items():
            assert hashlib.sha256((HERE/p).read_bytes()).hexdigest()==h, 'SOURCE_CHANGED '+p
        for p,h in freeze['input_sha256'].items():
            assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h, 'INPUT_CHANGED '+p
        for args in commands:
            left=45-(time.monotonic()-started)
            if left<=0:raise TimeoutError('TOTAL_WALL_CAP')
            p=subprocess.run([sys.executable,*args],cwd=HERE,env=os.environ.copy(),
                             capture_output=True,text=True,timeout=left)
            receipt['commands'].append(dict(args=args,returncode=p.returncode,stdout=p.stdout,stderr=p.stderr))
            if p.returncode:raise RuntimeError('VALIDATION_COMMAND_FAILED '+args[0])
        receipt['status']='COMPLETED_INSPECT_SCIENTIFIC_RESULT'
    except Exception as e:
        receipt['status']='STOPPED_NO_RETRY';receipt['error']=repr(e)
    finally:
        after=resource.getrusage(resource.RUSAGE_CHILDREN)
        receipt.update(wall_seconds=time.monotonic()-started,CPU_seconds=after.ru_utime+after.ru_stime-before.ru_utime-before.ru_stime,
                       peak_child_RSS_KiB=after.ru_maxrss,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
        out.mkdir(exist_ok=True)
        (out/'process_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps({k:receipt[k] for k in ['status','wall_seconds','CPU_seconds','peak_child_RSS_KiB']}),flush=True)
        if receipt['status']!='COMPLETED_INSPECT_SCIENTIFIC_RESULT':sys.exit(1)


if __name__=='__main__':main()
