"""Replay the unchanged parent harness across resets with deterministic actions.

Its existing multi-episode explore loop is used only as a loop. Sampling is
disabled; no optimizer or controller definition is changed.
"""
import shutil
from pathlib import Path
from posture_episode_ppo import PosturePPO
original=PosturePPO.choose
def deterministic(self,obs,stochastic=True):return original(self,obs,False)
PosturePPO.choose=deterministic
import run_posture_episode_ppo as parent
if __name__=='__main__':
    parent.main()
    import sys
    out=Path(sys.argv[sys.argv.index('--output')+1]);shutil.copy2(__file__,out/'source'/Path(__file__).name)
