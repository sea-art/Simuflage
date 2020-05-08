
import math
import numpy as np
from scipy.constants import convert_temperature

# Electro-Migration related parameters
BETA = 2
ACTIVATIONENERGY = 0.48
BOLTZMANCONSTANT = 8.6173324 * 0.00001
CONST_JMJCRIT = 1500000
CONST_N = 1.1
CONST_ERRF = 0.88623
CONST_A0 = 30000  # cross section = 1um^2  material constant = 3*10^13

# Thermal model parameters
ENV_TEMP = 295  # room temperature
SELF_TEMP = 40  # self contribution
NEIGH_TEMP = 5  # neighbor contribution


def electro_migration(c_temp):
    temp = convert_temperature(c_temp, 'C', 'K')  # temp in Kelvin
    return (CONST_A0 * (np.power(CONST_JMJCRIT, (-CONST_N))) * np.exp(ACTIVATIONENERGY / (BOLTZMANCONSTANT * temp))) / CONST_ERRF