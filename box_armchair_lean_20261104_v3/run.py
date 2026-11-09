"""One bounded owner. Only two independent repeats after a clean center start."""
import fcntl,os,sys,subprocess,time,hashlib,json
from pathlib import Path
from audit import check
P=Path(__file__).resolve().parent;ROOT=P.parents[1];OUT=Path(sys.argv[1])
assert not OUT.exists();OUT.mkdir(parents=True)
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
locks=[]
for lock in (D/'contact_parallel_gpu_v1/transition.lock',D/'contact_ppo_scientific_owner.lock'):
    f=lock.open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
gpu='GPU-ab1950b0-01dd-0fab-407f-2d17936b27f4'
assert not subprocess.check_output(['nvidia-smi','-i',gpu,'--query-compute-apps=pid','--format=csv,noheader']).strip()
files=[*P.glob('*.py'),*P.glob('protocol.json'),ROOT/'experiments/furniture_perception_repair_20261029_v1/freeze.json']
(OUT/'freeze.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in files},indent=2))
frozen=json.loads((ROOT/'experiments/furniture_perception_repair_20261029_v1/freeze.json').read_text())
pick=['experiments/full_body_contact_trainer_4090_step_v5/'+x for x in ['common.py','audit.py','environment.py','policy.py']]
pick+=['experiments/furniture_perception_repair_20261029_v1/'+x for x in ['capture.py','perception.py','render_colliders.json']]
pick+=['experiments/furniture_test_20261028_v1/scene_adapter.py','experiments/furniture_validation_20261028_v2/contact_adapter.py','scripts/run_supported_postures.py']
for rel in pick:
    file=ROOT/rel
    assert hashlib.sha256(file.read_bytes()).hexdigest()==frozen[str(file)],rel
inputs=[ROOT/'runs/g1_kinematics_geometry_04/model.json',ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning/bank.npz',ROOT/'outputs/full_body_contact_imitation_v4/dataset/leaning/metadata.json',D.parent/'crouch_rest_bridge_01/balanced_standing10_01/static_inertial_model.json',*list((ROOT/'experiments/furniture_test_20261028_v1/scenes/myd1WSucAz_rear_back').glob('*'))]
(OUT/'inputs.json').write_text(json.dumps({str(f):hashlib.sha256(f.read_bytes()).hexdigest() for f in inputs if f.is_file()},indent=2))
for i in range(3):
    out=OUT/f'trial_{i}'
    with (OUT/f'trial_{i}.log').open('x') as log:
        subprocess.run(['timeout','-k','25','600','bash',str(ROOT/'scripts/run_supported_gpu0.sh'),str(P/'trial.py'),'--trial',str(i),'--output',str(out),'--device','cuda:0','--kit_args=--/renderer/activeGpu=0,--/renderer/multiGpu/enabled=false'],check=True,stdout=log,stderr=subprocess.STDOUT)
    if not check(out):break
