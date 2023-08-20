from .bm_matmul import BmMatmul
from .bm_topopt import BmTopopt
from .bm_wall_simple import BmWallSimple


def teneva_bm_get_various():
    Bms = []
    Bms.append(BmMatmul)
    Bms.append(BmTopopt)
    Bms.append(BmWallSimple)
    return Bms
