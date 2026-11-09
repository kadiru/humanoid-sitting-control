"""One full72 evaluation with corrected perception; no new training."""
import sys
from paths import ROOT,PRIOR
sys.path[:0]=[str(PRIOR),str(ROOT/'scripts')]
import run_training
import run_supported_transition
from perception import observe
run_supported_transition.observed_support=lambda model,K,T,depth,q,gravity,prior=None:observe(model,K,T,depth,q,gravity,'seat',prior)
if __name__=='__main__':run_training.main()
