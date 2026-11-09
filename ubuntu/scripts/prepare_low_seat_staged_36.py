"""Offline exact-input preparation of the staged candidate; no simulation."""
from pathlib import Path
import json,hashlib,datetime
import numpy as np
from humanoid_sitting.shared_frame import BodyModel
from low_seat_staged_planner_36 import plan,DURATION,LAND_START,LAND_END
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');OUT=D/'staged_preparation_36'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    OUT.mkdir(exist_ok=False);src=D/'normal_landing3s_06/candidate'
    tr=np.load(src/'trace.npz');ss=np.load(src/'sensor_inputs.npz')['sensors'];at=int(np.argmin(abs(tr['t']-1.)))
    mp=Path('runs/g1_kinematics_geometry_04/model.json');ip=D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json';cp=D.parent/'contact_effort_repair_01/plane_calibration.json';cfg=D/'normal_landing3s_guarded_09.json'
    m=BodyModel(mp);limits={j['name']:np.deg2rad(j['limits']) for j in read(mp)['joints'] if j['type']=='PhysicsRevoluteJoint'}
    fingers=[j for j,n in enumerate(m.names) if any('_'+s+'_' in n for s in ['zero','one','two','three','four','five','six'])];assert len(fingers)==14
    rows=[]
    for case in [0,1,2,3,6,7,10,11]:
        q=tr['q'][at,case];cmd=tr['command'][at,case];g=ss[0,10,case,270+69:270+72]
        assert np.array_equal(ss[0,10,case,270:293],q[:23])
        c,qs,a=plan(m,q,g,tr['geometry'][at,case],[tr['estimated_root_x'][at,case],0,0],cmd,read(cp),read(ip),read(cfg))
        assert np.array_equal(c[0],cmd) and np.array_equal(qs[0],q)
        assert np.array_equal(c[:,fingers],np.tile(cmd[fingers],(25,1)))
        jm=min(min(float((c[:,j]-limits[n][0]).min()),float((limits[n][1]-c[:,j]).min())) for j,n in enumerate(m.names[:23]));assert jm>=.03
        # Exact software first-order filter/cap replay only, not predicted physics.
        applied=cmd.astype(float).copy();records=[]
        for t in np.arange(0,DURATION+.0001,.02):
            u=t/DURATION*24;lo=min(int(u),23);f=min(1.,u-lo);ref=(1-f)*c[lo]+f*c[lo+1]
            before=applied.copy();applied[:23]+=np.clip(.1*(ref[:23]-applied[:23]),-.004,.004)
            assert abs(applied-before).max()<=.0040000001
            records.append(dict(elapsed=float(t),max_reference_command_gap=float(abs(ref[:23]-applied[:23]).max())))
        at_landing=min(records,key=lambda r:abs(r['elapsed']-LAND_START))
        row=dict(case=case,screen=a['staged_36']['screen'],command_joint_margin_rad=jm,fixed14_fingers=True,initial_command_exact=True,software_filter_gap_at_descent_start_rad=at_landing['max_reference_command_gap'],physical_success_claim=False,audit=a)
        rows.append(row);(OUT/f'case_{case}.json').write_text(json.dumps(row,indent=2));np.savez_compressed(OUT/f'path_{case}.npz',command=c,q=qs,root=np.array(a['root_path']))
        print(case,row['screen'],at_landing,flush=True)
    paths=[Path(__file__),Path('scripts/low_seat_staged_planner_36.py'),mp,ip,cp,cfg,src/'trace.npz',src/'sensor_inputs.npz']
    (OUT/'preparation.json').write_text(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,scope='Offline8-case actual1s sensor-input deterministic planner and command/filter checks. No physical support or contact success is certified. New first1s prefix;20mm hover,normal descent,previous-.35armoffset,pitch.9. .175nominal/.20applied planned research bounds; no material/physical-label change.',physics_ticks=0,ppo_updates=0,holdout=False,source_input_hashes={str(p.resolve()):sha(p) for p in paths},rows=rows),indent=2)+'\n')
if __name__=='__main__':main()
