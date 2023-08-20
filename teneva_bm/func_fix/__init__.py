from .bm_func_fix_piston import BmFuncFixPiston


def teneva_bm_get_func_fix():
    Bms = []
    Bms.append(BmFuncFixPiston)
    return Bms
