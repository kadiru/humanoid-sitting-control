import json,sys,time,os
from pathlib import Path
origin=float(sys.argv[1])
p=Path(__file__).parent/'backend_returned.json'
temp=p.with_suffix('.tmp')
temp.write_text(json.dumps(dict(origin=origin,elapsed_s=time.monotonic()-origin)))
os.replace(temp,p)
time.sleep(5)
