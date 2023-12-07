import numpy as np
from gekko import GEKKO


def vanderpol_oscillator(bm, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    # m.options.MAX_ITER = m
    m.time = np.linspace(0, 20, bm.d)

    # state variables
    x1 = m.Var(value=1)
    x2 = m.Var(value=0)
    x3 = m.Var(value=0)

    # control variable
    u = m.MV(value=0, lb=-0.75, ub=0.75)
    u.STATUS = 1

    m.Equation(x1.dt() == x2)
    m.Equation(x2.dt() == u * (1 - x1 ** 2) * x2 - x1)
    m.Equation(x2.dt() == x1 ** 2 + x2 ** 2 + u ** 2)

    final = m.Param(np.zeros(bm.d))
    final.value[-1] = 1
    m.Obj(x3 * final)

    m.solve(disp=False)
    control = u.VALUE
    obj = m.options.OBJFCNVAL
    return control, obj