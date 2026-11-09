import hashlib,json,os,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
OLD=D/'interior_after_minimax_20261023_v1'
O=D/'interior_multiplier_postmortem_20261023_v1'
CAP=45.
LIMIT=40*1024**3
def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4194304),b''):h.update(b)
 return h.hexdigest()
def write(p,x):
 with Path(p).open('x') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def check(origin):
 if time.monotonic()-origin>=CAP:raise TimeoutError('45-second total cap')
def admit(stage,origin):
 c=read(HERE/'contract.json');assert c['ready'] and c['read_only_postmortem'] and not c['candidate_admission_permitted']
 assert os.environ.get('MULTIPLIER_POSTMORTEM_STAGE')==stage
 a=read(O/'admission.json');assert a['origin']==origin and a['token']==os.environ.get('MULTIPLIER_POSTMORTEM_TOKEN')
 for p,h in c['input_sha256'].items():assert sha(p)==h,p;check(origin)
 for n,h in c['source_sha256'].items():assert sha(HERE/n)==h,n
 assert read(OLD/'audit.json')['passed'] and not read(OLD/'audit.json')['candidate_accepted']
 assert read(OLD/'numerical_checks.json')['finite'] and not read(OLD/'numerical_checks.json')['dual_signs']
 return c
def workers(excluded):
    result=[]
    for p in Path('/proc').glob('[0-9]*/cmdline'):
        if int(p.parent.name) in excluded: continue
        try: args=p.read_bytes().decode().split('\0')
        except (OSError, UnicodeError): continue
        if args and any(x in Path(args[0]).name.lower() for x in ['python','kit','isaac']):
            if any(x in ' '.join(args) for x in ['humanoid-sitting','low_seat','IsaacLab']):
                result.append(dict(pid=int(p.parent.name), args=args))
    return result
