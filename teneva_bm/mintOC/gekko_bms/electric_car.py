import numpy as np
from gekko import GEKKO


def electric_car(bm, m):
    m = GEKKO(remote=False)
    m.options.IMODE = 6
    m.options.SOLVER = 1
    # m.options.MAX_ITER = m
    m.time = np.linspace(0, 10, bm.d)

    # parameters
    K_r = 10     # Coefficient of reduction
    rho = 1.293  # Air density
    C_x = 0.4    # Aerodynamic coefficient
    S = 2        # Area in the front of the vehicle
    r = 0.33     # Radius of the wheel
    K_f = 0.03   # Constant representing the friction of the wheels on the road
    K_m = 0.27   # Coefficient of the motor torque
    R_m = 0.03   # Inductor resistance
    L_m = 0.05   # Inductance of the rotor
    M = 250      # Mass
    g = 9.81     # Gravity constant
    V_alim = 150 # Battery voltage
    R_bat = 0.05 # Resistance of the battery
    i_max = 150  # Max. value of x1
    # parameters' combinations
    k_1 = r / K_r
    k_2 = M * g * K_f
    k_3 = 0.5 * rho * S * C_x
    k_4 = M * r ** 2

    # state variables
    x1 = m.Var(value=0)
    x2 = m.Var(value=0)
    x3 = m.Var(value=0)
    x4 = m.Var(value=0)

    # control variable
    u_int = m.MV(value=0, lb=0, ub=1, integer=True)
    u_int.STATUS = 1
    u = m.Intermediate(u_int * 2 - 1)

    m.Equation(x1.dt() == (V_alim * u - R_m * x1 - K_m * x2) / L_m)
    m.Equation(x2.dt() == (K_m * x1 - k_1 * (k_2 + k_3 * (k_1 * x2) ** 2)) * K_r ** 2 / k_4)
    m.Equation(x3.dt() == k_1 * x2)
    m.Equation(x4.dt() == V_alim * u * x1 + R_bat * x1 ** 2)

    # objective
    final = m.Param(np.zeros(bm.d))
    final.value[-1] = 1
    m.Obj(x4 * final)

    # constraints
    m.Equation(m.abs(x3 - 100) * final < bm.constr_eps)
    m.Equation(-1 * (x1 + i_max) < bm.constr_eps)
    m.Equation(-1 * (i_max - x1) < bm.constr_eps)

    m.solve(disp=False)
    control_var = u.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj