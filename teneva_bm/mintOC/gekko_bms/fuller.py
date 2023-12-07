import numpy as np
from gekko import GEKKO


def fuller(bm, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    # m.options.MAX_ITER = m
    m.time = np.linspace(0, 10, bm.d)

    # state variables
    x1 = m.Var(value=0.01)
    x2 = m.Var(value=0)
    x3 = m.Var(value=0)

    # control variable
    w = m.MV(value=0, lb=0, ub=1, integer=True)
    w.STATUS = 1

    m.Equation(x1.dt() == x2)
    m.Equation(x2.dt() == 1 - 2 * w)
    m.Equation(x2.dt() == x1 ** 2)

    final = m.Param(np.zeros(bm.d))
    final.value[-1] = 1
    m.Obj(x3 * final)

    # constraints
    m.Equation(m.abs(x1 - 0.01) * final < bm.constr_eps)
    m.Equation(m.abs(x2) * final < bm.constr_eps)
    
    m.solve(disp=False)
    control = w.VALUE
    obj = m.options.OBJFCNVAL
    return control, obj