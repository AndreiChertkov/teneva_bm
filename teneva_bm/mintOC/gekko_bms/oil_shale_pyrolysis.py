import numpy as np
from gekko import GEKKO

def oil_shale_pyrolysis(d, m):
    m = GEKKO() # remote=False
    m.options.IMODE = 6
    m.options.SOLVER = 3
    m.options.MAX_ITER = m
    m.time = np.linspace(0, 1, d)
    
    # parameters
    # frequency factors
    a1 = np.exp(8.86)
    a2 = np.exp(24.25)
    a3 = np.exp(23.67)
    a4 = np.exp(18.75)
    a5 = np.exp(20.7)
    # activation energies
    b1 = 20.3
    b2 = 37.4
    b3 = 33.8
    b4 = 28.2
    b5 = 31.0
    # universal gas constant
    R = 1.9858775e-3
    
    # state variables
    x1 = m.Var(value=1) # kerogen
    x2 = m.Var(value=0) # pyrolytic bitumen
    x3 = m.Var(value=0) # oil and gas
    x4 = m.Var(value=0) # carbon residue
    
    # control variable
    T = m.MV(value=698.15, lb=698.15, ub=748.15)
    T.STATUS = 1
    T.DCOST = 0
    
    # final time
    tf = m.FV(value=1, lb=0.1, ub=20)
    tf.STATUS = 1

    k1 = m.Intermediate(a1 * m.exp(-b1 / (R * T)))
    k2 = m.Intermediate(a2 * m.exp(-b2 / (R * T)))
    k3 = m.Intermediate(a3 * m.exp(-b3 / (R * T)))
    k4 = m.Intermediate(a4 * m.exp(-b4 / (R * T)))
    k5 = m.Intermediate(a5 * m.exp(-b5 / (R * T)))
    
    m.Equation(x1.dt()/tf == -k1 * x1 - (k3 + k4 + k5) * x1 * x2)
    m.Equation(x2.dt()/tf == k1 * x1 - k2 * x2 + k3 * x1 * x2)
    m.Equation(x3.dt()/tf == k2 * x2 + k4 * x1 * x2)
    m.Equation(x4.dt()/tf == k5 * x1 * x2)
    
    final = m.Param(np.zeros(d))
    final.value[-1] = 1
    m.Obj(-x2 * final)
    m.solve(disp=False)

    control_var = T.VALUE
    obj = m.options.OBJFCNVAL
    return control_var, obj