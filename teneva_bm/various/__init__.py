from .bm_matmul import BmMatmul
from .bm_topopt import BmTopopt
from .bm_wall_simple import BmWallSimple


def teneva_bm_get_various():
    bms = []
    bms.append(BmMatmul)
    bms.append(BmTopopt)
    bms.append(BmWallSimple)

    return bms
