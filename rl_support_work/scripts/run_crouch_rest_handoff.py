"""Reference-only probes after the complete validated approach."""
import argparse,sys
import torch

p=argparse.ArgumentParser(add_help=False);p.add_argument('--handoff-seconds',type=float,default=12.)
handoff_args,rest=p.parse_known_args();sys.argv=[sys.argv[0]]+rest
import run_crouch_rest_probe as probe

class HandoffEnv(probe.ProbeEnv):
    def __init__(self,raw,args,*rest,**kwargs):
        assert args.mode in ['reference','fixed'], 'Development reference probe, not a training contract'
        super().__init__(raw,args,*rest,**kwargs)
    def _observe(self,force=False):
        enabled=self.planning_enabled
        if enabled:self.planning_enabled=bool((self.motion_time()>=handoff_args.handoff_seconds).any())
        try:super()._observe(force)
        finally:self.planning_enabled=enabled
    def close(self):
        super().close()
        from pathlib import Path
        import json
        (self.out/'handoff_config.json').write_text(json.dumps(vars(handoff_args),indent=2))
        (self.out/'source'/'run_crouch_rest_handoff.py').write_text(Path(__file__).read_text())

probe.progressive.ProgressiveEnv=HandoffEnv
if __name__=='__main__':probe.progressive.main()
