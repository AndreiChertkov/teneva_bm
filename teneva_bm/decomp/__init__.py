from .bm_decomp_ht import BmDecompHt
from .bm_decomp_peps import BmDecompPeps
from .bm_decomp_tt import BmDecompTt


def teneva_bm_get_decomp():
    Bms = []
    Bms.append(BmDecompHt)
    Bms.append(BmDecompPeps)
    Bms.append(BmDecompTt)
    return Bms
