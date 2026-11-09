"""Software invariants only: these fixtures are not robot demonstrations."""
import copy
import unittest
import numpy as np
from stages import RestCycle, Sensors, Stage
from surface_memory import SurfaceMemory, Localization
from action import apply
from cycle_audit import audit


def packet(**kw):
    d=dict(map_valid=True,tracking_ok=True,quiet=True,positioned=True,
           standing=True,clear_of_support=True,load_lower_bw=.15,load_upper_bw=.2)
    d.update(kw);return Sensors(**d)


class CoreTests(unittest.TestCase):
    def test_event_cycle_and_no_simulated_success_claim(self):
        c=RestCycle()
        for _ in range(1000):out=c.update(packet())
        self.assertEqual(c.stage,Stage.UNLOAD)
        self.assertFalse(out['physical_success'])
        for _ in range(125):out=c.update(packet(load_lower_bw=0.,load_upper_bw=.005))
        self.assertEqual(c.stage,Stage.DONE)
        self.assertFalse(out['physical_success'])
        self.assertEqual([e['stage'] for e in c.events],
                         ['observe','position','contact','transfer','hold','unload','stand','done'])

    def test_contact_spike_is_not_dwell(self):
        c=RestCycle()
        for _ in range(50):c.update(packet())
        self.assertEqual(c.stage,Stage.CONTACT)
        for _ in range(60):c.update(packet(load_lower_bw=None,load_upper_bw=None))
        c.update(packet())
        self.assertEqual(c.stage,Stage.CONTACT)
        for _ in range(24):c.update(packet())
        self.assertEqual(c.stage,Stage.TRANSFER)

    def test_unknown_load_cannot_unload_or_trigger_standing(self):
        c=RestCycle()
        for _ in range(1000):c.update(packet())
        for _ in range(30):out=c.update(packet(load_lower_bw=None,load_upper_bw=None))
        self.assertFalse(out['advance_primitive']);self.assertEqual(c.stage,Stage.UNLOAD)
        c.update(packet(load_lower_bw=0.,load_upper_bw=.005))
        self.assertEqual(c.stage,Stage.UNLOAD)

    def test_hold_interruption_does_not_accumulate(self):
        c=RestCycle()
        for _ in range(900):c.update(packet())
        c.update(packet(tracking_ok=False))
        for _ in range(99):c.update(packet())
        self.assertEqual(c.stage,Stage.ABORT)
        self.assertEqual(c.reason,'original20s_hold_not_completed')

    def test_contact_and_cycle_deadlines(self):
        c=RestCycle()
        for _ in range(1000):c.update(packet(load_lower_bw=None,load_upper_bw=None))
        self.assertEqual(c.stage,Stage.ABORT)
        c=RestCycle()
        for _ in range(2001):c.update(packet())
        self.assertEqual(c.stage,Stage.ABORT)
        self.assertEqual(c.reason,'complete_cycle_deadline')

    def test_standing_requires_whole_two_seconds(self):
        c=RestCycle()
        for _ in range(1000):c.update(packet())
        for _ in range(124):c.update(packet(load_lower_bw=0.,load_upper_bw=.005))
        self.assertEqual(c.stage,Stage.STAND)
        c.update(packet(load_lower_bw=0.,load_upper_bw=.005,standing=False))
        for _ in range(99):c.update(packet(load_lower_bw=0.,load_upper_bw=.005))
        self.assertEqual(c.stage,Stage.STAND)
        c.update(packet(load_lower_bw=0.,load_upper_bw=.005))
        self.assertEqual(c.stage,Stage.DONE)

    def test_invalid_load_aborts(self):
        for lo,hi in [(None,.1),(.1,None),(float('nan'),.2),(.4,.1)]:
            c=RestCycle();c.update(packet(load_lower_bw=lo,load_upper_bw=hi))
            self.assertEqual(c.stage,Stage.ABORT)

    def test_map_transform_not_false_freshness(self):
        p=np.array([[0.,0.,.5],[.1,0.,.5],[0.,.1,.5]])
        m=SurfaceMemory(p,[0,0,1],-.5,0.)
        T=np.eye(4);T[:3,:3]=[[1,0,0],[0,0,-1],[0,1,0]];T[:3,3]=[.2,.3,.4]
        r=m.estimate(10.,Localization(T,9.9,.001,.001,True))
        self.assertTrue(r['valid']);self.assertEqual(r['visual_age_s'],10.)
        self.assertLess(np.max(abs(r['points']@r['normal']+r['offset'])),1e-12)
        self.assertFalse(m.estimate(10.,Localization(T,9.,.001,0.,True))['valid'])
        self.assertFalse(m.estimate(10.,Localization(T,9.9,.001,0.,False))['valid'])
        self.assertFalse(m.estimate(10.,Localization(T,9.9,.004,0.,True))['valid'])
        T[0,0]=-1
        self.assertFalse(m.estimate(10.,Localization(T,9.9,0.,0.,True))['valid'])
        self.assertFalse(m.verify_observation(p+.004))
        self.assertFalse(m.estimate(10.,Localization(np.eye(4),10.,0.,0.,True))['valid'])

    def test_action_contract_and_invalid_inputs(self):
        p=np.zeros(37,np.float32);p[23:]=.12
        for _ in range(100):
            nxt=apply(p,np.ones(23)*2,np.ones(23)*-1,np.ones(23))
            self.assertTrue((abs(nxt-p)<=.003+2e-7).all())
            np.testing.assert_array_equal(nxt[23:],p[23:]);p=nxt
        bad=p.copy();bad[3]=np.nan
        with self.assertRaises(ValueError):apply(bad,np.zeros(23),-np.ones(23),np.ones(23))
        with self.assertRaises(ValueError):apply(p,np.zeros(23),np.ones(23),-np.ones(23))

    @staticmethod
    def record():
        n=1200
        return dict(q=np.zeros((n,37)),qd=np.zeros((n,37)),command=np.zeros((n,37)),desired=np.zeros((n,23)),
                    failure=np.zeros(n,bool),no_slip=np.ones(n,bool),legal=np.ones(n,bool),good=np.ones(n,bool),
                    rest_load_bw=np.r_[np.ones(1000)*.2,np.zeros(200)],pitch_error=np.zeros(n),gyro_y=np.zeros(n),
                    standing_height_error=np.zeros(n),time_s=.02*np.arange(1,n+1),
                    stage=np.array(['observe']*25+['position']*25+['contact']*25+['transfer']*25+['hold']*900+['unload']*25+['stand']*174+['done']),
                    initial_command=np.zeros(37),standing_q=np.zeros(37),lower=-np.ones(23),upper=np.ones(23))

    def test_audit_accepts_fixture_but_not_as_physical_evidence(self):
        r=audit(self.record(),'sitting',evidence_kind='synthetic_fixture')
        self.assertTrue(r['passed'],r);self.assertFalse(r['physical_success'])

    def test_audit_rejects_each_material_failure(self):
        edits=[('failure',10,True),('no_slip',10,False),('legal',10,False),('good',990,False),
               ('stage',50,'position'),('stage',900,'unload'),('command',(0,0),.004),
               ('command',(0,24),.001),('command',(100,0),.001),('q',(1150,0),.06),
               ('qd',(1150,0),.06),('rest_load_bw',1150,.02),('pitch_error',1150,np.nan),
               ('time_s',100,.1)]
        for k,i,v in edits:
            z=self.record();z[k][i]=v
            # Moving one phase boundary by one tick is permitted; insert a reversal instead.
            if k=='stage' and i==50:z[k][60]='observe'
            with self.subTest(field=k,index=i):self.assertFalse(audit(z,'sitting',evidence_kind='synthetic_fixture')['passed'])
        z=self.record();z['q'][750:1000,11:13]=np.deg2rad(83)
        self.assertFalse(audit(z,'upright_lean',evidence_kind='synthetic_fixture')['passed'])


if __name__=='__main__':unittest.main(verbosity=2)
