from pathlib import Path
import json,time,subprocess,os
D=Path(/data/humanoid-sitting-perception/low_seat_contact_01);R=Path(/home/kadir/humanoid-sitting)
while True:
 s=json.loads((D/status_touchdown_probe_178.json).read_text())
 if s[phase]==physics_complete_audit_pending:break
 if s[phase]==failed:raise RuntimeError(s)
 time.sleep(5)
subprocess.run([str(R/IsaacLab/.venv/bin/python),str(R/scripts/finalize_low_seat_touchdown_181.py)],cwd=R,check=True)
