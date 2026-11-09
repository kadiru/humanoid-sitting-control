"""CPU contracts: buffer indexing and unchanged upstream verdict identity."""
import json, sys, types
from pathlib import Path
import numpy as np

class Original:
    def __init__(self, result, support):
        self.result=result;self.support=support;self.frames=[]
    def __call__(self, raw):
        self.frames.append(dict(mapping_valid=False))
        return self.result

fake=types.ModuleType('contact_adapter');fake.MeshMetrics=Original;fake.array=lambda a:np.asarray(a).copy()
sys.modules['contact_adapter']=fake
from contact_recording import select_contact_arrays, RecordedMeshMetrics

values=[np.arange(7.)[:,None],np.arange(21.).reshape(7,3),np.ones((7,3)),np.zeros((7,1)),np.array([2]),np.array([3])]
selected=select_contact_arrays(values,8)
assert selected['tuple_components_0_to_3'][1]==values[1][3:5].tolist()
empty=values[:4]+[np.array([0]),np.array([-1])]
assert all(not x for x in select_contact_arrays(empty,8)['tuple_components_0_to_3'])
for bad in [values[:4]+[np.array([8]),np.array([0])],values[:4]+[np.array([2]),np.array([6])]]:
    try:select_contact_arrays(bad,8)
    except ValueError:pass
    else:raise AssertionError('invalid buffer admitted')

class Sim:
    def get_physics_step_count(self):return 104
    def get_physics_dt(self):return .005
sensor=types.SimpleNamespace(contact_view=types.SimpleNamespace(get_contact_data=lambda **kw:values),
                             cfg=types.SimpleNamespace(max_contact_data_count_per_prim=8))
raw=types.SimpleNamespace(sim=Sim(),scene={'contact':sensor})
support=types.SimpleNamespace(CONTACTS=[('contact','pelvis',True)])
result=({'force_w':np.ones((1,1,3)),'normal_w':np.array([[[0.,0.,1.]]]),
         'inside_patch':np.array([[False]]),'failure':np.array([True])},{'good':np.array([False])},{'forbidden':np.array([True])})
recorder=RecordedMeshMetrics(result,support)
assert recorder(raw) is result and recorder(raw) is result
assert len(recorder.frames)==2 and len(recorder.evidence)==1
assert recorder.frames[-1]['physics_step']==104 and recorder.evidence[0]['physics_clock_s']==.52
assert not result[1]['good'][0] and result[2]['forbidden'][0]
receipt=dict(passed=True,tests=['nonzero buffer offset','empty contact with sentinel index','saturation rejection',
                              'range rejection','unchanged upstream tuple and failure identity','deduplicated physical timestamp'],
             real_simulator_api_validated=False,simulation_steps=0,ppo_updates=0)
Path(__file__).with_name('contact_recording_check.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps(receipt))
