from .bm_odeoc_simple import BmOdeocSimple
from .bm_odeoc_simple_constr import BmOdeocSimpleConstr


def teneva_bm_get_odeoc():
    bms = []
    bms.append(BmOdeocSimple)
    bms.append(BmOdeocSimpleConstr)

    return bms
