import numpy as np
from gekko import GEKKO

def time_optimal_car(d, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    m.options.SOLVER = 1
    m.options.MAX_ITER = m
    m.time = np.linspace(0, 1, d)
    
    # state variables
    x1 = m.Var(value=0, ub=330, lb=0)
    x2 = m.Var(value=0, ub=33, lb=0)
    m.fix_final(x1, 300)
    m.fix_final(x2, 0)
    
    # control variable
    u = m.MV(integer=True, lb=-2, ub=1)
    u.STATUS = 1

    # final time
    tf = m.FV(value=500, lb=0.1)
    tf.STATUS = 1
    
    m.Equation(x1.dt()/tf == x2)
    m.Equation(x2.dt()/tf == u)

    m.Obj(tf)
    
    m.solve(disp=False)
    control_var = u.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj