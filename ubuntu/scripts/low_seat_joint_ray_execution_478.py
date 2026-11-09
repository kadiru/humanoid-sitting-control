"""NEW complete shared-direction execution; no472 optimizer continuation.
Uses its sealed INITIAL as an explicit research seed, with no repeated initial
root or finite-work loop. One exact affine-ray scalar replaces all472 groupwise
contractions. Original full-control/drop boxes and all actual gates remain.
"""
from low_seat_free_gap_execution_472 import *
from low_seat_joint_ray_476 import decode
SPEC=dict(SPEC)
SPEC.update(seed='Sealed472 INITIAL X/Y/J and exact drop state, explicitly inherited without repeating its completed root; no rejected trial used.',budget_justification='ONE new shared-affine-ray formulation, same24/four/120 bounded budget justified by18 half-reductions plus6 rejection opportunities; not a restart or extension of472.',normal_drop_map='Exact common affine ray shared with every load/p/free-gap direction;468 joint directed encoding retains exact pivot/endpoints. No independent group contractions.',joint_direction='Largest exact scalar in[0,1] meeting every target/drop and full linearized f/p/load step bound; actual dependent f/phase/hard constraints checked after new root.',prediction='Rate/window prediction evaluated on the SAME shared scaled direction, with unchanged half-improvement corrector target and acceptance ratio.',finite_map_derivative='Continuous unrounded interior zero-ray B only; quantized map not literally differentiable.')
class SharedRayRetraction(GapRetraction):
 def trial(self,a,tangent,folder):
  self.counts['finite_maps']+=1
  assert self.counts['finite_maps']<=SPEC['max_consecutive_load_roundings']
  target,drops,record=decode(self.d,a,tangent)
  (folder/'mapped.json').write_text(json.dumps(record))
  np.savez_compressed(folder/'mapped.npz',target=target,tangent=tangent)
  x=a['X']+np.einsum('nic,nc->ni',a['T'],target-a['target'])
  result=self.lift(x,target,folder)
  result.update(target=target,drops=drops,map_record=record)
  return result
