"""Data-free harness: prevents study reads, subprocesses and network calls."""
import hashlib
import io
import json
import pathlib
import sys
import time
import unittest

HERE=pathlib.Path(__file__).resolve().parent


def guard(event,args):
    if event=='open':
        p=str(args[0])
        if '/data/humanoid-sitting' in p or 'validated_low_seat_contact_01' in p:
            raise PermissionError('synthetic harness forbids scientific data')
    if event.startswith(('subprocess.','socket.','os.system','os.exec','os.posix_spawn')):
        raise PermissionError('synthetic harness forbids processes/network')


sys.addaudithook(guard)
stream=io.StringIO()
start=time.monotonic()
suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_*.py')
result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
elapsed=time.monotonic()-start
record=dict(classification='Synthetic streaming actual-hard assembly and return tests only',
            tests=result.testsRun,passed=result.wasSuccessful(),failures=len(result.failures),
            errors=len(result.errors),seconds=elapsed,study_data_reads=0,solver_calls=0,model_calls=0,
            scientific_launch_ready=False,
            files_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(HERE.glob('*.py'))})
number=1
while (HERE/f'synthetic_run_{number:02d}.json').exists():number+=1
(HERE/f'synthetic_run_{number:02d}.log').write_text(stream.getvalue())
(HERE/f'synthetic_run_{number:02d}.json').write_text(json.dumps(record,indent=2)+'\n')
print(stream.getvalue())
print(json.dumps(record))
sys.exit(0 if result.wasSuccessful() else 1)
