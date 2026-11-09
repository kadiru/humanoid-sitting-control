"""No scientific reads, subprocesses, network or optimizer/model entry points."""
import hashlib,io,json,pathlib,sys,time,unittest
HERE=pathlib.Path(__file__).resolve().parent
def guard(event,args):
    if event=='open':
        p=str(args[0])
        if '/data/humanoid-sitting' in p or 'validated_low_seat_contact_01' in p:
            raise PermissionError('Synthetic tests forbid scientific data')
    if event.startswith(('subprocess.','socket.','os.system','os.exec','os.posix_spawn')):
        raise PermissionError('Synthetic tests forbid process/network calls')
sys.addaudithook(guard)
for p in HERE.rglob('*.py'):compile(p.read_text(),str(p),'exec')
out=io.StringIO();start=time.monotonic()
suite=unittest.defaultTestLoader.discover(str(HERE),pattern='test_integration.py')
result=unittest.TextTestRunner(stream=out,verbosity=2).run(suite)
record=dict(classification='Synthetic stage-integration checks only',tests=result.testsRun,
    passed=result.wasSuccessful(),errors=len(result.errors),failures=len(result.failures),
    seconds=time.monotonic()-start,scientific_data_reads=0,solver_calls=0,model_calls=0,
    source_sha256={str(p.relative_to(HERE)):hashlib.sha256(p.read_bytes()).hexdigest()
                   for p in sorted(HERE.rglob('*.py'))})
i=1
while (HERE/f'synthetic_integration_{i:02d}.json').exists():i+=1
(HERE/f'synthetic_integration_{i:02d}.json').write_text(json.dumps(record,indent=2)+'\n')
(HERE/f'synthetic_integration_{i:02d}.log').write_text(out.getvalue())
print(out.getvalue());print(json.dumps({k:v for k,v in record.items() if k!='source_sha256'}))
sys.exit(0 if result.wasSuccessful() else 1)
