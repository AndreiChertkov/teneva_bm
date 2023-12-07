import numpy as np
from gekko import GEKKO


def bioreactor(bm, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    # m.options.MAX_ITER = m
    m.time = np.linspace(0, 48, bm.d)

    # parameters
    D = 0.15    # Dilution
    K_i = 22    # Rate coefficient
    K_m = 1.2   # Rate coefficient
    P_m = 50    # Rate coefficient
    Y_xs = 0.4  # Substrate to Biomass rate
    alpha = 2.2 # Linear slope
    beta = 0.2  # Linear intercept
    mu_m = 0.48 # Maximal growth rate

    # state variables
    x1 = m.Var(value=6.5)
    x2 = m.Var(value=12)
    x3 = m.Var(value=22)
    x4 = m.Var(value=0)

    # control variable
    x2_f = m.MV(value=28.7, lb=28.7, ub=40)
    x2_f.STATUS = 1

    mu = m.Intermediate(mu_m * (1 - x3 / P_m) * x2 / (K_m + x2 + x2 ** 2 / K_i))
    m.Equation(x1.dt() == (mu - D) * x1)
    m.Equation(x2.dt() == D * (x2_f - x2) - (mu / Y_xs) * x1)
    m.Equation(x3.dt() == -D * x3 + (alpha * mu + beta) * x1)
    m.Equation(x4.dt() == D * (x2_f - x3) ** 2)

    final = m.Param(np.zeros(bm.d))
    final.value[-1] = 1
    m.Obj(x4 * final)

    m.solve(disp=False)
    control_var = x2_f.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj