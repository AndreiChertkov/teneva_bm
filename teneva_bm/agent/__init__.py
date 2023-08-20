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
    Bms = []
    Bms.append(BmAgentAnt)
    Bms.append(BmAgentCheetah)
    Bms.append(BmAgentHuman)
    Bms.append(BmAgentHumanStand)
    Bms.append(BmAgentLake)
    Bms.append(BmAgentLander)
    Bms.append(BmAgentPendInv)
    Bms.append(BmAgentPendInvDouble)
    Bms.append(BmAgentReacher)
    Bms.append(BmAgentSwimmer)
    return Bms
