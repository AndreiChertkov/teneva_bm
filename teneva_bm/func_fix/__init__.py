from .bm_func_fix_biggs import BmFuncFixBiggs
from .bm_func_fix_cantrell import BmFuncFixCantrell
from .bm_func_fix_colville import BmFuncFixColville
from .bm_func_fix_dolan import BmFuncFixDolan
from .bm_func_fix_piston import BmFuncFixPiston


def teneva_bm_get_func_fix():
    Bms = []
    Bms.append(BmFuncFixBiggs)
    Bms.append(BmFuncFixCantrell)
    Bms.append(BmFuncFixColville)
    Bms.append(BmFuncFixDolan)
    Bms.append(BmFuncFixPiston)
    return Bms
