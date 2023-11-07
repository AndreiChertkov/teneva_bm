import numpy as np
from GEKKO.gekko import GEKKO


def hanging_chain(d, m): 
    m = GEKKO() # remote=False
    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.SOLVER = 3
    m.options.TIME_SHIFT = 0
    m.options.MAX_ITER = m
    m.time = np.linspace(0, 1, d)
    
    # parameters
    a = 1
    b = 3
    Lp = 4

    # state variables
    x1 = m.Var(value=a, lb=0, ub=10)
    x2 = m.Var(value=0, lb=0, ub=10)
    x3 = m.Var(value=0, lb=0, ub=10)

    # control variable
    u = m.MV(value=0, lb=-10, ub=20) # integer=True
    
    m.Equation(x1.dt() == u)
    m.Equation(x2.dt() == x1 * (1 + u ** 2) ** 0.5)
    m.Equation(x3.dt() == (1 + u ** 2) ** 0.5)
    
    final = m.Param(np.zeros(d))
    final.value[-1] = 1
    m.Obj(1000 * (x1 - b) ** 2 * final)
    m.Obj(1000 * (x3 - Lp) ** 2 * final)
    m.Obj(x2 * final)
    
    u.STATUS = 0
    m.solve(disp=False)

    u.STATUS = 1
    u.DCOST = 1e-3
    m.solve(disp=False)

    control_var = u.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj