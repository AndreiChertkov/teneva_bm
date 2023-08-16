__version__ = '0.7.4'


from .bm import Bm


from .agent import *
from .func import *
from .hs import *
from .odeoc import *
from .qubo import *
from .various import *


def teneva_bm_get():
    bms = []
    bms += teneva_bm_get_agent()
    bms += teneva_bm_get_func()
    bms += teneva_bm_get_hs()
    bms += teneva_bm_get_odeoc()
    bms += teneva_bm_get_qubo()
    bms += teneva_bm_get_various()
    return bms
