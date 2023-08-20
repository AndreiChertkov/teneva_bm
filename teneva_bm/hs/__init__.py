from .bm_hs_func001 import BmHsFunc001
from .bm_hs_func006 import BmHsFunc006


def teneva_bm_get_hs():
    Bms = []
    Bms.append(BmHsFunc001)
    Bms.append(BmHsFunc006)
    return Bms
