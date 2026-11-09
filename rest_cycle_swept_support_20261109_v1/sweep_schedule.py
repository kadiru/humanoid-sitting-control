"""Use the original .003rad/tick ceiling with exact inverse-filter margins."""
import numpy as np

def schedule(nodes,lower,upper):
    """Exact original inverse filter, including endpoint directional headroom."""
    current=np.asarray(nodes[0]['command']).copy();commands=[];desired=[];phase=[]
    for a,b in zip(nodes,nodes[1:]):
        target=np.asarray(b['command']);delta=target-current
        head=np.where(delta[:23]>=0,upper-.03-target[:23],target[:23]-lower-.03)
        if np.min(head)<0:raise ValueError('ENDPOINT_COMMAND_OUTSIDE_ORIGINAL_MARGIN')
        # For a linear segment, desired_last=target+9*delta/N. Check all ticks
        # independently afterwards. This retains the original.03rad margin.
        n=max(1,int(np.ceil(np.max(abs(delta))/(.003-1e-12))),int(np.ceil(np.max(9*abs(delta[:23])/np.maximum(head,1e-12)))))
        if a.get('normal') or b.get('normal'):
            n=max(n,int(np.ceil(abs(b['gap']-a['gap'])/.006/.02)))
        if n>1000 or len(commands)+n>1000:raise ValueError('FINITE20S_COMMAND_RETURN_CAP')
        for j in range(1,n+1):
            cmd=current+delta*j/n;previous=current+delta*(j-1)/n
            desired.append((previous[:23]+10*(cmd[:23]-previous[:23])).tolist())
            commands.append(cmd.tolist());phase.append(b['phase'])
        current=target
    return dict(commands=commands,desired=desired,phase=phase,seconds=len(commands)*.02)
