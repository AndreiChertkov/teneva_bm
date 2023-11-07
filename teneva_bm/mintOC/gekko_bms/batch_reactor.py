import numpy as np
from gekko import GEKKO


def batch_reactor(d, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.MAX_ITER = m
    m.time = np.linspace(0, 1, d)

    # state variables
    x1 = m.Var(value=1)
    x2 = m.Var(value=0)
    
    # control variable
    T = m.MV(value=398, lb=298, ub=398)
    T.STATUS = 1
    T.DCOST = 1e-6

    k1 = m.Intermediate(4e3 * m.exp(-2.5e3 / T))
    k2 = m.Intermediate(6.2e5 * m.exp(-5e3 / T))
    m.Equation(x1.dt() == -k1 * x1 ** 2)
    m.Equation(x2.dt() == k1 * x1 ** 2 - k2 * x2)

    final = m.Param(np.zeros(d))
    final.value[-1] = 1
    m.Obj(-x2 * final)

    m.solve(disp=False)
    control_var = T.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj