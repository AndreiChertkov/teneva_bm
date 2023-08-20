from .bm_odeoc_simple import BmOdeocSimple
from .bm_odeoc_simple_constr import BmOdeocSimpleConstr


def teneva_bm_get_odeoc():
    Bms = []
    Bms.append(BmOdeocSimple)
    Bms.append(BmOdeocSimpleConstr)
    return Bms
