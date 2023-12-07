import numpy as np
from gekko import GEKKO


def lotka_volterra(bm, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    m.options.SOLVER = 1
    # m.options.MAX_ITER = m
    m.time = np.linspace(0, 12, bm.d)

    # parameters
    c0 = 0.4 
    c1 = 0.2

    # state variables
    x0 = m.Var(value=0.5, lb=0)
    x1 = m.Var(value=0.7, lb=0)
    x2 = m.Var(value=0.0, lb=0)

    # control variable
    w = m.MV(value=0, lb=0, ub=1, integer=True)
    w.STATUS = 1
    
    m.Equation(x0.dt() == x0 - x0 * x1 - c0 * x0 * w)
    m.Equation(x1.dt() == - x1 + x0 * x1 - c1 * x1 * w)
    m.Equation(x2.dt() == (x0 - 1) ** 2 + (x1 - 1) ** 2)
    
    final = m.Param(np.zeros(bm.d))
    final.value[-1] = 1
    m.Obj(x2 * final)
    m.solve(disp=False)

    control_var = w.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj