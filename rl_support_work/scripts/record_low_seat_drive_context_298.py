"""Read-only authored drive/source context; no environment, query or physics."""
from pathlib import Path
from pxr import Usd,UsdPhysics
import hashlib,json,datetime,shutil
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01');R=Path('/home/kadir/humanoid-sitting');O=D/'drive_context_298'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);asset=D/'asset_contact_47/g1.usd';st=Usd.Stage.Open(str(asset));rows=[]
 for p in st.Traverse():
  if p.IsA(UsdPhysics.RevoluteJoint):
   drive=UsdPhysics.DriveAPI(p,'angular');rows.append(dict(path=str(p.GetPath()),joint=p.GetName(),type=str(drive.GetTypeAttr().Get()),stiffness=float(drive.GetStiffnessAttr().Get()),damping=float(drive.GetDampingAttr().Get()),max_force=float(drive.GetMaxForceAttr().Get())))
 assert len(rows)==37
 files=['IsaacLab/source/isaaclab/isaaclab/actuators/actuator_collection.py','IsaacLab/source/isaaclab/isaaclab/actuators/actuator_control.py','IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/actuator_control.py','IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation.py','IsaacLab/source/isaaclab_physx/isaaclab_physx/assets/articulation/articulation_data.py','scripts/run_low_seat_drive_diagnostic_246.py']
 (O/'source').mkdir();hashes={str(asset):sha(asset),str(Path(__file__).resolve()):sha(__file__)}
 for k,name in enumerate(files):p=R/name;hashes[str(p)]=sha(p);shutil.copy2(p,O/'source'/f'{k}_{p.name}')
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope=__doc__,authored_drives=rows,source_sha256=hashes,physics_ticks=0,environment_created=False,live_backend_readback=False,controller_changed=False,physical_admission=False,
 observations=[
 'Preserved USD ankle-roll drive type is force, authored stiffness100/damping0. Recorded259 internal joint-property buffers and implicit kernel inputs instead have runtime160/4; configuration changes these values during initialization. This expected source distinction is not evidence of a failed setter.',
 'Installed ActuatorCollection resolves properties; ArticulationActuatorControl.write_resolved_joint_properties forwards implicit stiffness/damping to PhysX articulation writers. Writers update internal buffers and call root_view.set_dof_stiffnesses/set_dof_dampings using backend joint order.',
 'Recorded actuator_metadata uses data.joint_stiffness/joint_damping property buffers. These are not an independent post-configuration backend readback. Direct live drive-type/gain/friction and incoming-joint-force capture remain unavailable in these recordings.',
 '296 equivalent boundary moments are inferred under explicitly stated finite-interval/inertia assumptions. They cannot be substituted for motor effort or fed to control. The approximate telemetry discrepancy does not identify a setter, gain, solver or sensor defect.',
 'Before relying on290 as a response predictor, a separately scoped passive backend property/incoming-joint-wrench recorder could test this missing correspondence. It must prove ordering, frames, private storage, public-cache preservation and exact global replay before interpretation. Do not repeat an unchanged prior diagnostic or automatically launch physics.'
 ],primary_documentation=[dict(url='https://nvidia-omniverse.github.io/PhysX/physx/5.8.0/docs/Articulations.html',accessed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),summary='NVIDIA documents articulation drives as implicit end-of-step position/velocity constraints, distinguishes force and acceleration drive units, and recommends incoming joint force instead of deprecated joint solver force. This describes the API/model, not the effective runtime state of259.')])
 (O/'context.json').write_text(json.dumps(out,indent=2));(O/'SHA256.json').write_text(json.dumps({str(p.relative_to(O)):sha(p) for p in O.rglob('*') if p.is_file()},indent=2));print(json.dumps(dict(joints=len(rows),ankles=[r for r in rows if 'ankle_roll' in r['joint']],physics_ticks=0)),flush=True)
if __name__=='__main__':main()
