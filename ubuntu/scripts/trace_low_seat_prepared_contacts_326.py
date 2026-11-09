"""Read-only contact-parameter provenance; no simulation or fitted internal state."""
from pathlib import Path
import json, hashlib, datetime, re
from pxr import Usd, UsdGeom, UsdPhysics

D = Path('/data/humanoid-sitting-perception/low_seat_contact_01')
S = D/'prepared_contact_source_326'
O = D/'prepared_contact_trace_326'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
read = lambda p: json.loads(Path(p).read_text())
def write(p, x): p.write_text(json.dumps(x, indent=2, allow_nan=False)+'\n')

def main():
    O.mkdir(exist_ok=False)
    source = read(S/'source_manifest.json')
    for row in source['sources']:
        assert sha(S/row['local']) == row['sha256']
    stage = Usd.Stage.Open(str(D/'asset_contact_47/g1.usd'))
    names = ['physxRigidBody:cfmScale', 'physxCollision:contactOffset',
             'physxCollision:restOffset', 'physxScene:solveArticulationContactLast',
             'physxMaterial:frictionCombineMode', 'physxScene:enableStabilization']
    authored = []
    bodies, collisions = [], []
    prims = list(stage.Traverse())
    for prim in prims:
        if prim.HasAPI(UsdPhysics.RigidBodyAPI): bodies.append(str(prim.GetPath()))
        if prim.HasAPI(UsdPhysics.CollisionAPI): collisions.append(str(prim.GetPath()))
        for name in names:
            attr = prim.GetAttribute(name)
            if attr and attr.HasAuthoredValueOpinion():
                authored.append(dict(path=str(prim.GetPath()), attribute=name,
                                     value=str(attr.Get()), provenance='preserved asset; not live stage'))
    live = read(D/'incoming_diagnostic_v2_303/live_stage_settings.json')
    live_hits = {name: [dict(path=r['path'], value=r['attributes'][name])
                       for r in live['settings'] if name in r['attributes']] for name in names}
    assert not any(live_hits[n] for n in names[:5])
    assert live_hits[names[5]] == [dict(path='/physicsScene', value='False')]
    assert [r for r in authored if r['attribute']==names[5]] == [dict(
        path='/physicsScene',attribute=names[5],value='True',provenance='preserved asset; not live stage')]
    scene = next(r['attributes'] for r in live['settings'] if r['path']=='/physicsScene')
    tokens = {
        'schema.usda': ['float physxRigidBody:cfmScale = 0.025',
            'float physxCollision:contactOffset = -inf','float physxCollision:restOffset = -inf',
            'bool physxScene:solveArticulationContactLast = false',
            'uniform token physxMaterial:frictionCombineMode = "average"'],
        'artiConstraintPrep2.cu': ['b.cfm = artic.cfms[linkID]',
            'const PxReal cfm = PxMax(b0.cfm, b1.cfm)',
            'const float penetration = separation - restDistance',
            '(1.f / (unitResponse + cfm))',
            'scaledBias = isSeparated ? -invDt : -invDtWithBiasCoefficient',
            'frictionPatch, prevFrictionPatches, fAnchor, prevFrictionAnchors, prevFrictionIndices'],
        'PxsMaterialCombiner.h': ['PxMax(mat0Data.getFrictionCombineMode(), mat1Data.getFrictionCombineMode())'],
        'unitree.py': ['max_depenetration_velocity=1.0',
                      'solver_position_iteration_count=8, solver_velocity_iteration_count=4'],
        'run_resting_posture.py': ['static_friction=.8,dynamic_friction=.6'],
    }
    evidence = []
    for file, snippets in tokens.items():
        text = (S/file).read_text()
        for snippet in snippets:
            assert snippet in text, (file,snippet)
            line = text[:text.index(snippet)].count('\n')+1
            evidence.append(dict(file=file, line=line, token=snippet))
    # Restrict absence claims to explicitly enumerated sources, not all possible runtime writes.
    overrides = []
    for file in ['task_config.py','unitree.py','run_resting_posture.py','run_supported_postures.py']:
        for i, line in enumerate((S/file).read_text().splitlines(),1):
            if re.search(r'cfm[_:]?[Ss]cale|contact_offset\s*=|rest_offset\s*=|solve_articulation_contact_last\s*=', line):
                overrides.append(dict(file=file,line=i,text=line))
    rows = [
      dict(parameter='Per-link CFM', supplied='Installed schema fallback cfmScale=0.025; absent as authored opinion in preserved robot asset.',
           source_rule='Pinned forwardDynamic2.cu scales cfmScale by inverse length, then maximum linear articulated impulse response; artiConstraintPrep2 uses max(link CFM).',
           live='Neither live cfmScale nor prepared link/row CFM captured.', recoverable=False),
      dict(parameter='Contact / rest offsets', supplied='Installed schema sentinel -inf delegates contact offset to shape extent and rest offset to simulation; schema documentation says rigid-body rest default zero. Preserved asset has no authored offsets.',
           source_rule='PxShape documents sums of shape offsets; articulation preparation uses separation minus pair restDistance.',
           live='Actual collider offsets, pair restDistance and prepared separation unavailable. Default text is not a measured zero offset.', recoverable=False),
      dict(parameter='Normal row response and bias', supplied='G1 config max_depenetration_velocity=1 m/s; stage state and response evolve inside solver.',
           source_rule='Articulation branch uses reciprocal(response+CFM); separation sign selects bias scaling; impulse is bounded nonnegative. Restitution/compliance branches remain distinct.',
           live='Per-stage response, bias clamp, prepared separation, target velocity and accumulated motion absent.', recoverable=False),
      dict(parameter='Friction material and patch', supplied='Seat/floor config static/dynamic 0.8/0.6; installed material defaults 0.5/0.5, schema combine average. Existing study robot 0.5/0.5 unchanged.',
           source_rule='Pinned combiner chooses higher-priority pair mode; patch preparation uses cached anchors and scales coefficients for two anchors. Query points do not identify patch rows.',
           live='Captured scene patch friction, offset threshold '+scene['physxScene:frictionOffsetThreshold']+' m and correlation distance '+scene['physxScene:frictionCorrelationDistance']+' m. Pair combined coefficients/patch IDs/anchors/broken flags absent.', recoverable=False),
      dict(parameter='Ordering / iterations', supplied='Installed solveArticulationContactLast fallback false. Configured articulation 8 position / 4 velocity.',
           source_rule='Pinned GPU core has ordering branches and effective island iteration counts.',
           live='Captured authored TGS, 8/4 and external-forces-every-iteration true; effective counts/traversal/contact-last backend value absent.', recoverable=False),
      dict(parameter='Asset versus instantiated stage', supplied='Preserved source asset authors enableStabilization=True.',
           source_rule='Instantiation and configuration can override asset opinions.',
           live='Recorded instantiated scene enableStabilization=False. This is a provenance counterexample, not a changed setting or cause claim.', recoverable=True),
    ]
    inventory = dict(prim_count=len(prims), rigid_body_count=len(bodies), collision_count=len(collisions),
        meters_per_unit=UsdGeom.GetStageMetersPerUnit(stage), authored=authored,
        rigid_bodies=bodies,collisions=collisions,live_hits=live_hits,
        searched_config_overrides=overrides,source_asset_sha256=sha(D/'asset_contact_47/g1.usd'),
        live_stage_sha256=sha(D/'incoming_diagnostic_v2_303/live_stage_settings.json'))
    results = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        scope='Source/configuration/asset provenance only; no physics, new trajectory or fitted internal quantities.',
        source_commit=source['pinned_commit'], installed_extension='110.3.2', binary_equivalence_proven=False,
        rows=rows, evidence=evidence, inventory=inventory,
        exact_internal_stages_recoverable_from_saved_evidence=False,
        physical_admission=False,repair_complete=False,physics_ticks=0,optimizer_updates=0,
        conclusions=[
          'Defaults and source algorithms do not supply the uncaptured effective contact rows or internal iteration history.',
          'CFM is not an established cause of the raw-velocity/pose discrepancy. No CFM, offset, friction, ordering or iteration setting was changed or fitted.',
          'Endpoint contact separation and force cannot be substituted into a prepared-row update to recover actual stage velocity.',
          'A new complete approach remains possible, but no specific alternate motion is admitted by this trace. Missing internal states are not an impossibility proof.',
          'Keep current raw quiet/support guards and measured-sensor control; no predicted or conditional velocity replacement.'
        ])
    write(O/'results.json',results);write(O/'asset_inventory.json',inventory)
    write(O/'SHA256.json',{p.name:sha(p) for p in O.iterdir() if p.is_file()})
    write(S/'SHA256.json',{p.name:sha(p) for p in S.iterdir() if p.is_file() and p.name!='SHA256.json'})
    print(json.dumps(dict(prims=len(prims),bodies=len(bodies),collisions=len(collisions),authored=authored,
        evidence=len(evidence),source_files=len(source['sources']),overrides=overrides,physical_admission=False)))
if __name__=='__main__': main()
