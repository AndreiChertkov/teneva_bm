import numpy as np
from GEKKO.gekko import GEKKO


def lotka_volterra(d, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    m.options.NODES = 3
    m.options.SOLVER = 1
    m.options.MV_TYPE = 0
    m.options.MAX_ITER = m
    # m.solver_options = ['minlp_gap_tol 0.001',
    #                     'minlp_maximum_iterations 10000',
    #                     'minlp_max_iter_with_int_sol 100',
    #                     'minlp_branch_method 1',
    #                     'minlp_integer_tol 0.001',
    #                     'minlp_integer_leaves 0',
    #                     'minlp_maximum_iterations 200']
    m.time = np.linspace(0, 12, d)

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
    
    final = m.Param(np.zeros(d))
    final.value[-1] = 1
    m.Obj(x2 * final)
    m.solve(disp=False)

    control_var = w.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj