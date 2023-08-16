from .bm_agent_ant import BmAgentAnt
from .bm_agent_cheetah import BmAgentCheetah
from .bm_agent_human import BmAgentHuman
from .bm_agent_human_stand import BmAgentHumanStand
from .bm_agent_lake import BmAgentLake
from .bm_agent_lander import BmAgentLander
from .bm_agent_pend_inv import BmAgentPendInv
from .bm_agent_pend_inv_double import BmAgentPendInvDouble
from .bm_agent_reacher import BmAgentReacher
from .bm_agent_swimmer import BmAgentSwimmer


def teneva_bm_get_agent():
    bms = []
    bms.append(BmAgentAnt)
    bms.append(BmAgentCheetah)
    bms.append(BmAgentHuman)
    bms.append(BmAgentHumanStand)
    bms.append(BmAgentLake)
    bms.append(BmAgentLander)
    bms.append(BmAgentPendInv)
    bms.append(BmAgentPendInvDouble)
    bms.append(BmAgentReacher)
    bms.append(BmAgentSwimmer)

    return bms
