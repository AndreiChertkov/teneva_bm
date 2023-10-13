__version__ = '0.8.4'


from .bm import Bm
from .bm import BmBudgetOverException


from .agent import *
from .decomp import *
from .func import *
from .func_fix import *
from .hs import *
from .odeoc import *
from .qubo import *
from .qubo_fix import *
from .various import *


def teneva_bm_get(name=None, d=None, is_func=None, is_opti_max=None,
                  with_constr=None, with_cores=None, with_show=None,
                  with_render=None, with_max=None, with_min=None):
    Bms = []
    Bms += teneva_bm_get_agent()
    Bms += teneva_bm_get_decomp()
    Bms += teneva_bm_get_func()
    Bms += teneva_bm_get_func_fix()
    Bms += teneva_bm_get_hs()
    Bms += teneva_bm_get_odeoc()
    Bms += teneva_bm_get_qubo()
    Bms += teneva_bm_get_qubo_fix()
    Bms += teneva_bm_get_various()

    Bms_out = []
    for Bm in Bms:
        if name is not None:
            if name.lower() in Bm.__name__.lower():
                return Bm
            continue

        bm = Bm()

        if d is not None and not bm.check_args(d=d):
            continue
        if is_func is not None and bm.is_func != is_func:
            continue
        if is_opti_max is not None and bm.is_opti_max != is_opti_max:
            continue
        if with_constr is not None and bm.with_constr != with_constr:
            continue
        if with_cores is not None and bm.with_cores != with_cores:
            continue
        if with_show is not None and bm.with_show != with_show:
            continue
        if with_render is not None and bm.with_render != with_render:
            continue
        if with_max is not None and (bm.y_max_real is None) == with_max:
            continue
        if with_min is not None and (bm.y_min_real is None) == with_min:
            continue

        Bms_out.append(Bm)

    return Bms_out
