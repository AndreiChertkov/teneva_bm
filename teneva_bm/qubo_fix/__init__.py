from .bm_qubo_fix_knap10 import BmQuboFixKnap10
from .bm_qubo_fix_knap20 import BmQuboFixKnap20
from .bm_qubo_fix_knap50 import BmQuboFixKnap50
from .bm_qubo_fix_knap80 import BmQuboFixKnap80
from .bm_qubo_fix_knap100 import BmQuboFixKnap100


def teneva_bm_get_qubo_fix():
    Bms = []
    Bms.append(BmQuboFixKnap10)
    Bms.append(BmQuboFixKnap20)
    Bms.append(BmQuboFixKnap50)
    Bms.append(BmQuboFixKnap80)
    Bms.append(BmQuboFixKnap100)
    return Bms
