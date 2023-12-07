from .batch_reactor import BmBatchReactor
from .bioreactor import BmBioreactor
from .catalyst_mixing import BmCatalystMixing
from .double_tank import BmDoubleTank
from .electric_car import BmElectricCar
from .fuller import BmFuller
from .hanging_chain import BmHangingChain
from .lotka_volterra import BmLotkaVolterra
from .oil_shale_pyrolysis import BmOilShalePyrolysis
from .vanderpol_oscillator import BmVanderpolOscillator


def teneva_bm_get_mintOC():
    Bms = []
    Bms.append(BmBatchReactor)
    Bms.append(BmBioreactor)
    Bms.append(BmCatalystMixing)
    Bms.append(BmDoubleTank)
    Bms.append(BmElectricCar)
    Bms.append(BmFuller)
    Bms.append(BmHangingChain)
    Bms.append(BmLotkaVolterra)
    Bms.append(BmOilShalePyrolysis)
    Bms.append(BmVanderpolOscillator)
    return Bms