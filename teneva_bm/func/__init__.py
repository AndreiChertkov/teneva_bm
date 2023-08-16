from .bm_func_ackley import BmFuncAckley
from .bm_func_alpine import BmFuncAlpine
from .bm_func_dixon import BmFuncDixon
from .bm_func_exp import BmFuncExp
from .bm_func_griewank import BmFuncGriewank
from .bm_func_michalewicz import BmFuncMichalewicz
from .bm_func_piston import BmFuncPiston
from .bm_func_qing import BmFuncQing
from .bm_func_rastrigin import BmFuncRastrigin
from .bm_func_rosenbrock import BmFuncRosenbrock
from .bm_func_schaffer import BmFuncSchaffer
from .bm_func_schwefel import BmFuncSchwefel


def teneva_bm_get_func():
    Bms = []
    Bms.append(BmFuncAckley)
    Bms.append(BmFuncAlpine)
    Bms.append(BmFuncDixon)
    Bms.append(BmFuncExp)
    Bms.append(BmFuncGriewank)
    Bms.append(BmFuncMichalewicz)
    Bms.append(BmFuncPiston)
    Bms.append(BmFuncQing)
    Bms.append(BmFuncRastrigin)
    Bms.append(BmFuncRosenbrock)
    Bms.append(BmFuncSchaffer)
    Bms.append(BmFuncSchwefel)

    return Bms
