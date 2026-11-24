from contact_corridor_path import *
import json
def main():
    src=P/'corner_traction_16/evaluation'
    assert sha(src/'trajectory.npz')==read(src/'independent_audit.json')['files']['trajectory.npz']
    z=np.load(src/'trajectory.npz');j=14;assert z['good'][-250:,j].all()
    mat=np.load(P/'measured_corner_path_30.npz')['motion']
    ph=np.load(src/'corridor_feedback.npz')['data'][:,j,1]
    ix=np.r_[np.full(40,40),np.nan_to_num(ph,nan=40)].astype(int)
    np.savez_compressed(P/'intercept_landing_refs.npz',donor_desired=z['desired'][:,j],donor_J=mat[ix,84:].reshape(1000,12,6))
    opts={6:[[.007,-.003,.0005,.030,.027,0],[.008,0,.001,.032,.035,0],[.011,-.003,.001,.035,.045,0],[.013,0,.001,.040,.042,0],[.017,.003,.001,.03,.065,0],[.014,-.005,.001,.034,.06,0],[.009,.002,.001,.018,.05,.015],[.009,-.002,.001,.047,.025,-.015]],
          21:[[0,0,0,0,0,0],[.002,-.002,0,.018,0,0],[0,.004,0,-.025,.015,0],[-.003,0,0,0,-.024,0],[.005,0,0,.02,.026,0],[0,-.006,0,.030,0,0],[.003,.004,0,-.018,-.03,0],[-.005,-.004,0,.026,-.025,0]]}
    rows=[];routes={}
    for i,listings in opts.items():
        for a in listings:
            c=dict(start=422 if i==6 else 225,ramp=106 if i==6 else 80,pose=a,slow=305,end=405,pace=.65,trigger=.010,load=.088)
            rows.append(dict(asset_id=i,config=c));routes.setdefault(str(i),c)
    write(P/'intercept_contact_trials.json',dict(routes=routes,diagnostic_rows=rows))
if __name__=='__main__':main()

