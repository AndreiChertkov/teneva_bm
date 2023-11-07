import numpy as np
from gekko import GEKKO


def double_tank(d, m):
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
    m.time = np.linspace(0, 10, d)

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
    
    final = m.Param(np.zeros(d))
    final.value[-1] = 1
    m.Obj(x3 * final)

    m.solve(disp=False)
    control_var = sigma.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj