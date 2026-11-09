"""Regression for a subclass restoring raw.step after base initialization."""
from types import SimpleNamespace
import numpy as np
from capture_teacher import attach_on_first_step
from common import write
from pathlib import Path


def exercise(deferred, controls):
    calls=[];records=[];attachments=[]
    class Raw:
        def step(self, command):
            calls.append(command.copy())
            return command
    def attach(env):
        attachments.append(env.constructed)
        original=env.raw.step
        def observe(command):
            before=env.tick
            result=original(command)
            assert result is command
            records.append((before,command.copy()))
            return result
        env.raw.step=observe
    class Base:
        def __init__(self,raw):
            self.raw=raw;self.constructed=False;self.tick=0
            self.episode_length_buf=np.zeros(2,int);self.warming=np.zeros(2,int)
            raw.step(np.array([-1.,-1.]))  # startup must not enter dataset
            if deferred:attach_on_first_step(self,attach)
            else:attach(self)
        def step(self, action):
            for _ in range(controls):
                command=np.array([self.tick,self.tick+1.],float)
                result=self.raw.step(command);assert result is command
                self.tick+=1;self.episode_length_buf+=1
            return action
    class Concrete(Base):
        def __init__(self,raw):
            saved=raw.step
            def settling(command):return saved(command)
            raw.step=settling
            try:super().__init__(raw)
            finally:raw.step=saved  # the actual sitting constructor's overwrite
            self.constructed=True
    env=Concrete(Raw());action=object()
    for _ in range(1000//controls):assert env.step(action) is action
    assert len(calls)==1001
    if deferred:
        assert attachments==[True]
        assert len(records)==1000
        assert [r[0] for r in records]==list(range(1000))
        np.testing.assert_array_equal(np.stack([r[1] for r in records]),np.stack(calls[1:]))
        env.raw.step=lambda x:x
        try:env.step(action)
        except AssertionError as e:assert str(e)=='RECORDER_REPLACED_AFTER_ATTACHMENT'
        else:raise AssertionError('Overwritten recorder was not detected')
    else:assert len(records)==0
    return len(records)


if __name__=='__main__':
    result=dict(passed=True,old_hook_samples=exercise(False,5),
                sitting_samples=exercise(True,5),leaning_samples=exercise(True,1),
                commands_and_returns_unchanged=True,startup_excluded=True,
                later_hook_overwrite_rejected=True,simulator_runs=0)
    write(Path(__file__).with_name('attachment_tests.json'),result)
    print(result)
