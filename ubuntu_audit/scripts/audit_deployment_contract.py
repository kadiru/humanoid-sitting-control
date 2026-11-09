"""Check that the policy boundary cannot accept prohibited state channels."""
import ast,json,hashlib,argparse,dataclasses
from pathlib import Path
import torch
from humanoid_sitting.deployment import SensorPacket,pack,DeploymentStudent
p=argparse.ArgumentParser();p.add_argument('--output',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args();out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
source=Path(__file__).resolve().parents[1]/'sitting_task/src/humanoid_sitting/sim_deployment_sensor.py';tree=ast.parse(source.read_text());reads=sorted({n.attr for n in ast.walk(tree) if isinstance(n,ast.Attribute) and isinstance(n.value,ast.Name) and n.value.id=='d'})
allowed=['applied_torque','joint_pos','joint_vel','projected_gravity_b','root_ang_vel_b'];assert reads==allowed,reads
fields=[torch.zeros(2,d) for d in [23,23,23,3,3,23,10,1]];packet=SensorPacket(*fields);x=pack(packet);assert x.shape==(2,109)
rejected=[]
for name in ['contact_forces','root_position','root_linear_velocity','center_of_mass','simulation_time','ground_truth_geometry']:
 try:SensorPacket(**(dataclasses.asdict(packet)|{name:torch.zeros(2,1)}))
 except TypeError:rejected.append(name)
 else:raise AssertionError(name)
m,_=DeploymentStudent.load(a.checkpoint);state=None
with torch.inference_mode():
 for _ in range(5):actions,state=m(x[:,None,:],state);assert actions.shape==(2,1,23) and torch.isfinite(actions).all()
result=dict(effective_neural_channels=107 if m.mask_perception_status else 109,status_channels_used_for_motion=not m.mask_perception_status,physical_sensor_surrogate_reads=reads,actor_channels=109,actor_outputs=23,foot_feedback_channels=0,prohibited_packet_fields_rejected=rejected,sensor_adapter_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),verified=True,scope='Software channel boundary only; no physical sensor or motor calibration claim')
(out/'contract_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
