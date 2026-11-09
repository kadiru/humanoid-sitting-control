"""Commands are derived at each constrained pose; no command-space resampling."""
import numpy as np


def rolling_root(rolling,theta,touch_pitch,touch_position):
    root=np.eye(4);root[:3,:3]=rolling.rotation(theta)
    root[:3,3]=(np.asarray(touch_position)+rolling.translation(theta,np.zeros(3))
                -rolling.translation(touch_pitch,np.zeros(3)))
    return root


def encode(nodes,lower,upper):
    commands=np.asarray([n['command'] for n in nodes],float)
    if commands.ndim!=2 or commands.shape[1]!=37 or not 2<=len(commands)<=645 or not np.isfinite(commands).all():
        raise ValueError('FINITE_ORIGINAL_TIME_WINDOW_REQUIRED')
    delta=np.diff(commands,axis=0)
    if np.max(abs(delta))>.003+1e-12:raise ValueError('ORIGINAL_COMMAND_SPEED_EXCEEDED')
    if not np.array_equal(commands[:,23:],np.broadcast_to(commands[0,23:],commands[:,23:].shape)):
        raise ValueError('FIXED_FINGERS_REQUIRED')
    desired=commands[:-1,:23]+10*delta[:,:23]
    if not np.isfinite(desired).all() or np.any(desired<np.asarray(lower)+.03) or np.any(desired>np.asarray(upper)-.03):
        raise ValueError('ORIGINAL_INVERSE_FILTER_BOX_EXCEEDED')
    return dict(commands=commands[1:].tolist(),desired=desired.tolist(),
                phase=[n['phase'] for n in nodes[1:]],seconds=(len(commands)-1)*.02)


def assemble(nodes,lower,upper,association):
    returning=list(reversed(nodes))
    forward=encode(nodes,lower,upper);backward=encode(returning,lower,upper)
    if forward['seconds']+1.5>=14.4 or backward['seconds']+2.5>20:
        raise ValueError('ORIGINAL_QUALIFICATION_TIME_RESERVE')
    return dict(nodes=nodes,return_nodes=returning,forward=forward,backward=backward,
        nominal_rest_settle_s=forward['seconds']+1.5,nominal_return_settle_s=backward['seconds']+2.5,
        rest_load=nodes[-1]['load'],rest_model_effort_ratio=nodes[-1]['model_effort_ratio'],
        association=association,runtime_load_observer_admitted=False,physical_success=False)


def bind_cycle_stages(plan,hover_node,touch_node):
    """Bind the controller to existing geometry; change no numeric trajectory."""
    nodes=plan['nodes'];backward=plan['return_nodes']
    if not 0<hover_node<touch_node<len(nodes)-1:raise ValueError('ORDERED_STAGE_NODES_REQUIRED')
    if abs(nodes[hover_node]['gap']-.012)>1e-10 or abs(nodes[touch_node]['gap'])>1e-10:
        raise ValueError('HOVER_AND_TOUCH_GEOMETRY_REQUIRED')
    release_node=len(nodes)-1-hover_node
    if abs(backward[release_node]['gap']-.012)>1e-10:raise ValueError('RETURN_HOVER_GEOMETRY_REQUIRED')
    loaded=next((i for i in range(touch_node+1,len(nodes)-1) if nodes[i]['load']>=.07),None)
    if loaded is None:raise ValueError('QUALIFIED_LOAD_BOUNDARY_REQUIRED')
    nodes[hover_node]['normal']=True;nodes[touch_node]['normal']=True
    backward[release_node]['normal']=True
    plan['cycle_boundaries']=dict(hover_node=hover_node,touch_node=touch_node,
                                  loaded_node=loaded,release_node=release_node)
    return plan
