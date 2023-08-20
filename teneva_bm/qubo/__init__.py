from .bm_qubo_knap_quad import BmQuboKnapQuad
from .bm_qubo_maxcut import BmQuboMaxcut
from .bm_qubo_mvc import BmQuboMvc


def teneva_bm_get_qubo():
    Bms = []
    Bms.append(BmQuboKnapQuad)
    Bms.append(BmQuboMaxcut)
    Bms.append(BmQuboMvc)
    return Bms
