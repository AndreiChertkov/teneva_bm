from .bm_func_ackley import BmFuncAckley
from .bm_func_alpine import BmFuncAlpine
from .bm_func_chung import BmFuncChung
from .bm_func_dixon import BmFuncDixon
from .bm_func_exp import BmFuncExp
from .bm_func_griewank import BmFuncGriewank
from .bm_func_michalewicz import BmFuncMichalewicz
from .bm_func_pathological import BmFuncPathological
from .bm_func_pinter import BmFuncPinter
from .bm_func_powell import BmFuncPowell
from .bm_func_qing import BmFuncQing
from .bm_func_rastrigin import BmFuncRastrigin
from .bm_func_rosenbrock import BmFuncRosenbrock
from .bm_func_salomon import BmFuncSalomon
from .bm_func_schaffer import BmFuncSchaffer
from .bm_func_schwefel import BmFuncSchwefel
from .bm_func_sphere import BmFuncSphere
from .bm_func_squares import BmFuncSquares
from .bm_func_trid import BmFuncTrid
from .bm_func_trigonometric import BmFuncTrigonometric
from .bm_func_wavy import BmFuncWavy
from .bm_func_yang import BmFuncYang


def teneva_bm_get_func():
    Bms = []
    Bms.append(BmFuncAckley)
    Bms.append(BmFuncAlpine)
    Bms.append(BmFuncChung)
    Bms.append(BmFuncDixon)
    Bms.append(BmFuncExp)
    Bms.append(BmFuncGriewank)
    Bms.append(BmFuncMichalewicz)
    Bms.append(BmFuncPathological)
    Bms.append(BmFuncPinter)
    Bms.append(BmFuncPowell)
    Bms.append(BmFuncQing)
    Bms.append(BmFuncRastrigin)
    Bms.append(BmFuncRosenbrock)
    Bms.append(BmFuncSalomon)
    Bms.append(BmFuncSchaffer)
    Bms.append(BmFuncSchwefel)
    Bms.append(BmFuncSphere)
    Bms.append(BmFuncSquares)
    Bms.append(BmFuncTrid)
    Bms.append(BmFuncTrigonometric)
    Bms.append(BmFuncWavy)
    Bms.append(BmFuncYang)
    return Bms
