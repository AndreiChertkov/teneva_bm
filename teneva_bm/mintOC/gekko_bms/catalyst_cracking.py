import numpy as np
from gekko import GEKKO


def catalyst_cracking(d, m):
    assert d == 3, 'd should be 3'

    m = GEKKO(remote=False)
    m.options.IMODE = 6
    m.options.MAX_ITER = m
    m.time = np.arange(21)

    # parameters
    z1 = m.Param(np.array([
        1.0000, 0.8105, 0.6208, 0.5258, 0.4345, 0.3903, 0.3342, 
        0.3034, 0.2735, 0.2405, 0.2283, 0.2071, 0.1669, 0.1530, 
        0.1339, 0.1265, 0.1200, 0.0990, 0.0870, 0.0770, 0.0690
    ]))
    z2 = m.Param(np.array([
        0.0000, 0.2000, 0.2886, 0.3010, 0.3215, 0.3123, 0.2716, 
        0.2551, 0.2258, 0.1959, 0.1789, 0.1457, 0.1198, 0.0909, 
        0.0719, 0.0561, 0.0460, 0.0280, 0.0190, 0.0140, 0.0100
    ]))
    
    # state variables
    x1 = m.Var(value=1)
    x2 = m.Var(value=1)
    
    # control variables
    theta = [m.FV(value=0, lb=0, ub=10)] * 3
    for t in theta:
        t.STATUS = 1

    m.Equation(x1.dt() == -(theta[0] + theta[2]) * x1 ** 2)
    m.Equation(x2.dt() == theta[0] * x1 ** 2 - theta[1] * x2)

    m.Obj((x1 - z1) ** 2 + (x2 - z2) ** 2)

    m.solve(disp=False)
    control_var = [t.VALUE[0] for t in theta]
    obj = m.options.OBJFCNVAL
    return control_var, obj