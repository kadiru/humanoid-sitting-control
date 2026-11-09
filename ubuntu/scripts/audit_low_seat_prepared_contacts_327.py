"""Independent scope/provenance audit of326; no model or simulator execution."""
from pathlib import Path
import json,hashlib,datetime,re
from pxr import Usd,UsdPhysics,Sdf
D=Path('/data/humanoid-sitting-perception/low_seat_contact_01')
S=D/'prepared_contact_source_326';T=D/'prepared_contact_trace_v2_326';O=D/'prepared_contact_audit_327'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 O.mkdir(exist_ok=False);checks=0
 def check(x,label):
  nonlocal checks
  assert x,label
  checks+=1
 for folder in [S,T,D/'prepared_contact_trace_326',D/'prepared_asset_scope_recovery_326']:
  for file,h in read(folder/'SHA256.json').items():check(sha(folder/file)==h,str(folder/file))
 r=read(T/'results.json');a=r['inventory'];stage=Usd.Stage.Open(str(D/'asset_contact_47/g1.usd'))
 # Independent recursive child traversal explicitly enters instance proxies.
 prims=[]
 def visit(p):
  for child in p.GetFilteredChildren(Usd.TraverseInstanceProxies()):
   prims.append(child);visit(child)
 visit(stage.GetPseudoRoot())
 check(len(prims)==a['prim_count']==378,'expanded prim coverage')
 check(sum(p.IsInstanceProxy() for p in prims)==a['instance_proxy_count']==208,'proxy coverage')
 for api,key,count in [(UsdPhysics.RigidBodyAPI,'rigid_bodies',44),(UsdPhysics.CollisionAPI,'collisions',38)]:
  paths=sorted(str(p.GetPath()) for p in prims if p.HasAPI(api))
  check(paths==sorted(a[key]) and len(paths)==count,key)
 names=list(a['live_hits'])
 authored=[]
 for p in prims:
  for name in names:
   attr=p.GetAttribute(name)
   if attr and attr.HasAuthoredValueOpinion():authored.append((str(p.GetPath()),name,str(attr.Get())))
 check(sorted(authored)==sorted((x['path'],x['attribute'],x['value']) for x in a['authored']),'all expanded authored opinions')
 live=read(D/'incoming_diagnostic_v2_303/live_stage_settings.json')
 for name in names:
  hits=[{'path':v['path'],'value':v['attributes'][name]} for v in live['settings'] if name in v['attributes']]
  check(hits==a['live_hits'][name],name+' live capture coverage')
 check(sha(D/'asset_contact_47/g1.usd')==a['source_asset_sha256'],'asset source hash')
 check(sha(D/'incoming_diagnostic_v2_303/live_stage_settings.json')==a['live_stage_sha256'],'live source hash')
 # Sdf reads actual schema defaults independently of the trace's string tokens.
 layer=Sdf.Layer.FindOrOpen(str(S/'schema.usda'))
 expected={'/PhysxRigidBodyAPI.physxRigidBody:cfmScale':.025,
  '/PhysxCollisionAPI.physxCollision:contactOffset':float('-inf'),
  '/PhysxCollisionAPI.physxCollision:restOffset':float('-inf'),
  '/PhysxSceneAPI.physxScene:solveArticulationContactLast':False,
  '/PhysxMaterialAPI.physxMaterial:frictionCombineMode':'average'}
 defaults={}
 for path,value in expected.items():
  attr=layer.GetAttributeAtPath(path);check(attr is not None,path)
  v=attr.default
  if isinstance(value,float) and value==.025:check(abs(v-value)<1e-9,path+' float32 schema default')
  else:check(v==value,path+' default')
  defaults[path]=str(v)
 for evidence in r['evidence']:
  line=(S/evidence['file']).read_text().splitlines()[evidence['line']-1]
  check(evidence['token'] in line,'source token and line')
 for row in read(S/'source_manifest.json')['sources']:
  check(sha(S/row['local'])==row['sha256'],'pinned/copy source')
 # Check relevant articulation branch, not the separate rigid-body-only preparation.
 art=(S/'artiConstraintPrep2.cu').read_text();forward=(S/'forwardDynamic2.cu').read_text()
 for token in ['b.cfm = artic.cfms[linkID]','cfm = PxMax(b0.cfm, b1.cfm)',
               'penetration = separation - restDistance','1.f / (unitResponse + cfm)']:
  check(token in art,'articulation '+token)
 for token in ['links[linkID].cfmScale * invLengthScale',
               'PxMax(response.column[0][3], PxMax(response.column[1][4], response.column[2][5]))']:
  check(token in forward,'CFM source '+token)
 check(not r['physical_admission'] and not r['repair_complete'] and r['physics_ticks']==0,'no admission')
 check(not r['exact_internal_stages_recoverable_from_saved_evidence'],'unavailable retained')
 out=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),passed=True,checks=checks,
  expanded_prims=378,instance_proxies=208,rigid_bodies=44,collision_prims=38,
  schema_defaults=defaults,original_326_scope_passed=False,expanded_v2_scope_passed=True,
  exact_internal_stage_reconstruction=False,physical_admission=False,repair_complete=False,
  limitations=['Source/configuration provenance audit, not a dynamic prediction or internal-stage capture.',
               'Float32 schema-default comparison is a representation check; no physical or controller bound changed.'])
 (O/'independent_audit.json').write_text(json.dumps(out,indent=2)+'\n')
 (O/'SHA256.json').write_text(json.dumps({'independent_audit.json':sha(O/'independent_audit.json')},indent=2)+'\n')
 print(json.dumps(out))
if __name__=='__main__':main()
