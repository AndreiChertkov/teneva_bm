import numpy as np
from gekko import GEKKO


def hanging_chain(bm, m): 
    m = GEKKO() # remote=False
    m.options.IMODE = 6
    # m.options.MAX_ITER = m
    m.time = np.linspace(0, 1, bm.d)

    # state variables
    x1 = m.Var(value=1, lb=0, ub=10)
    x2 = m.Var(value=0, lb=0, ub=10)
    x3 = m.Var(value=0, lb=0, ub=10)

    # control variable
    u = m.MV(value=0, lb=-10, ub=20) # integer=True
    
    m.Equation(x1.dt() == u)
    m.Equation(x2.dt() == x1 * (1 + u ** 2) ** 0.5)
    m.Equation(x3.dt() == (1 + u ** 2) ** 0.5)
    
    final = m.Param(np.zeros(bm.d))
    final.value[-1] = 1
    m.Obj(x2 * final)

    # constraints
    m.Equation(m.abs(x1 - 3) * final < bm.constr_eps)
    m.Equation(m.abs(x3 - 4) * final < bm.constr_eps)
    m.Equation(-1 * x1 < bm.constr_eps)
    m.Equation(-1 * x2 < bm.constr_eps)
    m.Equation(-1 * x3 < bm.constr_eps)
    m.Equation(-1 * (10 - x1) < bm.constr_eps)
    m.Equation(-1 * (10 - x2) < bm.constr_eps)
    m.Equation(-1 * (10 - x3) < bm.constr_eps)
    
    u.STATUS = 0
    m.solve(disp=False)

    u.STATUS = 1
    u.DCOST = 1e-3
    m.solve(disp=False)

    control_var = u.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj