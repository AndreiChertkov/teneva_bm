import numpy as np
from gekko import GEKKO


def double_tank(bm, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    m.options.SOLVER = 1
    # m.options.MAX_ITER = m
    m.time = np.linspace(0, 10, bm.d)

    # parameters
    k1 = 2
    k2 = 3

    # state variables
    x1 = m.Var(value=2)
    x2 = m.Var(value=2)
    x3 = m.Var(value=0)

    # control variable
    sigma=m.MV(value=0, lb=0, ub=1, integer=True)
    sigma.STATUS = 1

    m.Equation(x1.dt() == sigma + 1 - m.sqrt(x1))
    m.Equation(x2.dt() == m.sqrt(x1) - m.sqrt(x2))
    m.Equation(x3.dt() == k1 * (x2 - k2) ** 2)
    
    final = m.Param(np.zeros(bm.d))
    final.value[-1] = 1
    m.Obj(x3 * final)

    m.solve(disp=False)
    control_var = sigma.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj