from .bm_hs_func001 import BmHsFunc001
from .bm_hs_func006 import BmHsFunc006


def teneva_bm_get_hs():
    bms = []
    bms.append(BmHsFunc001)
    bms.append(BmHsFunc006)

    return bms
