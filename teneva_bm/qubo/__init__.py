from .bm_qubo_knap_det import BmQuboKnapDet
from .bm_qubo_knap_quad import BmQuboKnapQuad
from .bm_qubo_maxcut import BmQuboMaxcut
from .bm_qubo_mvc import BmQuboMvc


def teneva_bm_get_qubo():
    bms = []
    bms.append(BmQuboKnapDet)
    bms.append(BmQuboKnapQuad)
    bms.append(BmQuboMaxcut)
    bms.append(BmQuboMvc)

    return bms
