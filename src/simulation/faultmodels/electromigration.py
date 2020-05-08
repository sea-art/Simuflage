#!/usr/bin/env python

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


def electro_migration(temperature):
    """ Generates the scaling value for the Weibull distribution based on Black's equation.

    :param temperature: float - temperature of component in Celsius
    :return: float - Scale parameter for the Weibull distribution based on the temperature
    """
    temp = convert_temperature(temperature, 'C', 'K')  # temp in Kelvin
    return (CONST_A0 * (np.power(CONST_JMJCRIT, (-CONST_N))) * np.exp(ACTIVATIONENERGY / (BOLTZMANCONSTANT * temp))) / CONST_ERRF
