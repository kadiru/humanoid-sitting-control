"""Command-index adapter for a newly admitted contact path.

Use only after the full current-camera path and loaded observer have passed
independent admission. In particular, the previous narrow calibration is NOT
admitted here. This module has packet-level tests, not a physical qualification.
"""
from contact_gate import ContactGate

class ContactExecutor:
    def __init__(self, plan, *, hover_index, contact_index,
                 current_path_admitted, loaded_observer_admitted):
        if not current_path_admitted or not loaded_observer_admitted:
            raise ValueError('CURRENT_PATH_AND_LOADED_OBSERVER_ADMISSION_REQUIRED')
        self.desired=plan['forward']['desired']
        if not 0<hover_index<contact_index<len(self.desired):
            raise ValueError('CONTACT_BOUNDARIES_OUTSIDE_PATH')
        self.index=0
        self.gate=ContactGate(hover_index,contact_index)
    def command(self, packet, previous_command):
        verdict=self.gate.update(self.index,packet)
        # Holding the existing command preserves the original filter state;
        # a pause does not consume a desired waypoint or skip a path sample.
        desired=list(previous_command[:23])
        if verdict['advance'] and self.index<len(self.desired):
            desired=list(self.desired[self.index]);self.index+=1
        return desired,dict(verdict,path_index=self.index,path_complete=self.index==len(self.desired))
