"""Bind each successful demonstration to its raw controls, sensors, and GPU log."""
from pathlib import Path
import sys,re,json,hashlib
import numpy as np

root=Path(sys.argv[1])
base=Path('/home/kadir/humanoid-sitting')
reset=np.load(base/'outputs/full_body_contact_imitation_v4/dataset/leaning/bank.npz')['reset_command'][0]
rows=[]
for index,dy in enumerate([0,.015,-.015]):
    folder=root/f'trial_{index}'
    def read(name):return json.loads((folder/name).read_text())
    def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
    a=read('independent_audit.json');manifest=read('manifest.json');log=(root/f'trial_{index}.log').read_text()
    assert re.search(r'NVIDIA GeForce RTX 4090\s+\| Yes: 0',log)
    assert re.search(r'Skipping NVIDIA graphics device.*RTX 4070',log)
    assert manifest['lateral_start_m']==dy and a['qualifying_lean'] and a['minimum_final_torso_bw']>=.05
    for name,expected in a['source_hashes'].items():assert expected==digest(folder/name)
    row=a['row'];sensor=read('sensor_audit.json')
    cap=np.load(folder/'frame_0.npz')
    assert len(cap['selected_pixels'])>=20 and cap['geometry'][11]==1 and read('frame_0.json')['actor_truth_input']==False
    with np.load(folder/'trajectory.npz') as z:
        command=np.vstack([reset,z['command'][:,0]])
        assert np.max(abs(np.diff(command[:,:23],axis=0)))<=.0030002
        assert (command[:,23:]==reset[23:]).all()
        assert np.isfinite(command).all()
        f=z['force'][-250:,0];n=z['physical_normal_w'][-250:,0]
        fn=(f*n).sum(-1);active=np.linalg.norm(f,axis=2)>.01*z['weight'][-250:]
        use=np.where(active,np.linalg.norm(f-fn[:,:,None]*n,axis=2)/(.6*np.maximum(fn,1e-7)),0)
        assert use.max()<=.9 and z['loaded_slip'].max()<=.02
        rows.append(dict(seed=manifest['seed'],initial_lateral_offset_mm=1000*dy,recorded_pelvis_y_at_start_m=float(cap['root_position'][1]),
            duration_s=row['duration_s'],final_hold_s=row['final_hold_s'],minimum_final_torso_bw=a['minimum_final_torso_bw'],
            max_final_traction_utilization=float(use.max()),max_episode_loaded_slip_m_s=float(z['loaded_slip'].max()),slip_ticks=row['slip_ticks'],
            illegal_ticks=row['illegal_ticks'],initial_selected_pixels=len(cap['selected_pixels']),gpu='RTX 4090',auditor_sha256=digest(folder/'independent_audit.json')))
print(json.dumps(rows,indent=2))
(root/'repeatability.json').write_text(json.dumps(dict(verified=True,scope='Three full standing starts, one box armchair, one frozen camera-guided controller.',trials=rows),indent=2)+'\n')
