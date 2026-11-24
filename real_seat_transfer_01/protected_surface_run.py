from dual_landing_run import P,read,launch_trial
import sys
def launch(name,config,half):
    p=read(P/config); configs=list(p['routes'].values())+[s['config'] for s in p.get('diagnostic_rows',[])]
    paths=sorted({c['trajectory'] for c in configs if 'trajectory' in c})
    extras=('surface_load_motion.py','surface_load_run.py','protected_surface_run.py','protected_surface_motion.py','intercept_landing_refs.npz','intercept_prepare.py','height_bridge_plan.py','native_skin_bridge.py',*paths)
    launch_trial(name,p,half,worker='protected_surface_worker.py',extras=extras)
if __name__=='__main__':
    name,config,*half=sys.argv[1:];launch(name,config,int(half[0]) if half else None)

