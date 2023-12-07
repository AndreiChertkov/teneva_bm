import numpy as np
from gekko import GEKKO


def catalyst_mixing(bm, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    m.options.SOLVER = 1
    # m.options.MAX_ITER = m
    m.time = np.linspace(0, 1, bm.d)

    # state variables
    x1 = m.Var(value=1)
    x2 = m.Var(value=0)
    
    # control variable
    w = m.MV(value=0, lb=0, ub=1, integer=True)
    w.STATUS = 1

    m.Equation(x1.dt() == w * (10 * x2 - x1))
    m.Equation(x2.dt() == w * (x1 - 10 * x2) - (1 - w) * x2)

    final = m.Param(np.zeros(bm.d))
    final.value[-1] = 1
    m.Obj((-1 + x1 + x2) * final)

    m.solve(disp=False)
    control_var = w.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj