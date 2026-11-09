"""CPU integration fixtures: data must survive a shutdown that exits Python."""
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from capture_teacher import save_capture, save_before_close


def main():
    tests=[]
    with tempfile.TemporaryDirectory() as d:
        out=Path(d)
        rows=[{'tick':np.full(2,t),'q_pre':np.full((2,37),t/1000.)} for t in range(1000)]
        reset={'q':np.zeros((2,37))}
        meta={'physics_steps_at_capture_install':600,'physics_steps_at_last_control':4600}
        job={'arguments':['--output',str(out)]}
        calls=[]
        def exiting_close():
            assert (out/'demonstration_capture.json').is_file()
            calls.append('runtime_exit')
            raise SystemExit(0)
        raw=SimpleNamespace(close=exiting_close)
        def flush():
            calls.append('save')
            save_capture(rows,meta,reset,job)
        save_before_close(raw,flush)
        for _ in range(2):
            try:raw.close()
            except SystemExit as e:assert e.code==0
        assert calls==['save','runtime_exit','runtime_exit']
        with np.load(out/'demonstrations_50hz.npz') as a:
            assert a['q_pre'].shape==(1000,2,37)
            np.testing.assert_array_equal(a['tick'][:,0],np.arange(1000))
        assert json.loads((out/'demonstration_capture.json').read_text())['saved_before_simulator_close']
        assert not list(out.glob('*.tmp*'))
        tests.append('Valid1000-sample capture persisted before an exiting runtime; double close does not rewrite')
    for mode in ['short','nonfinite','wrong_tick']:
        with tempfile.TemporaryDirectory() as d:
            out=Path(d);rows=[{'tick':np.full(2,t),'q_pre':np.zeros((2,37))} for t in range(1000)]
            if mode=='short':rows.pop()
            elif mode=='nonfinite':rows[10]['q_pre'][0,0]=np.nan
            else:rows[1]['tick'][0]=7
            calls=[];raw=SimpleNamespace(close=lambda:calls.append('close'))
            save_before_close(raw,lambda:save_capture(rows,meta,reset,{'arguments':['--output',str(out)]}))
            try:raw.close()
            except AssertionError:pass
            else:raise AssertionError('Invalid capture accepted')
            assert not calls and not (out/'demonstration_capture.json').exists()
            tests.append(mode+' rejected before success marker and shutdown')
    result={'passed':True,'tests':tests,'simulator_steps':0,'ppo_updates':0,'runtime_capture_validation':'not executed'}
    print(json.dumps(result))
    return result


if __name__=='__main__':
    result=main()
    Path(__file__).with_name('capture_shutdown_tests.json').write_text(json.dumps(result,indent=2)+'\n')
