#!/usr/bin/env python

""" The integrator contains all the functionalities that the simulator will run.

This file should describe how the Simulator elements are integrated with eachother in order
to run the simulation.
"""

import copy
import os

import numpy as np
from abc import ABC, abstractmethod

from simulation.elements.thermals import Thermals
from simulation.elements.agings import Agings
from simulation.elements.components import Components

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class AbsIntegrator(ABC):
    """ Abstract class for the integrator."""
    @abstractmethod
    def step(self, *args):
        pass

    @abstractmethod
    def print_status(self, timestep):
        pass

    @abstractmethod
    def log_timestep(self, timestep):
        pass


class Integrator(AbsIntegrator):
    """ This class should be edited when adding new elements or changing simulation functionality."""
    def __init__(self, design_point, policy):
        """ Integrator is used to change/add functionality to the simulator.

        All elements of the simulator are required to work together, this class achieves this collaboration between
        simulator classes. More information is given at the CONTRIBUTING.md file.

        :param design_point: Designpoint object representing a system to evaluate.
        """
        capacities, power_uses, max_temps, comp_loc_map, app_map = design_point.to_numpy()

        # Simulation variables
        self._components = Components(capacities, power_uses, comp_loc_map, app_map, policy)
        self._thermals = Thermals(self._components.workload, max_temps, comp_loc_map, self._components.alive_components)
        self._agings = Agings(self._components.alive_components, self._thermals.temps, self._components.workload)
        self._timesteps = 0

        self._reset_params = [copy.deepcopy(self._components),
                              copy.deepcopy(self._thermals),
                              copy.deepcopy(self._components.alive_components)]

    @property
    def timesteps(self):
        """ Getter function for the timesteps instance variable.

        :return: integer - representing the amount of timesteps already taken in the simulator.
        """
        return self._timesteps

    def reset(self):
        """ Resets the simulator back to default. Resamples all random variables.

        :return: None
        """
        self._components = copy.deepcopy(self._reset_params[0])
        self._thermals = copy.deepcopy(self._reset_params[1])
        self._agings = Agings(self._reset_params[2], self._thermals.temps, self._components.workload)
        self._timesteps = 0

    def step(self):
        """ Evaluate the simulator by one timestep.

        This function contains all integration aspects regarding the functionality per timestep of the simulator.
        Since these applications are ought to use variables from each other, this collaboration is implemented here.
        For more information, see CONTRIBUTING.md.

        :return: Boolean indicating if a core has failed this iteration.
        """
        self._thermals.step(self._components.comp_loc_map)

        remap_required = self._agings.step(self._components.alive_components, self._thermals.temps)

        system_ok = self._components.step(self._agings.cur_agings)
        # self._thermals.update_thermals(self._components.power_uses)

        print(self._components.workload)

        self._agings.resample_workload_changes(self._components.workload, self._thermals.temps)

        return system_ok

    def do_n_steps(self):
        """ Will skip all intermediate steps and will directly go to the timestep that an event occurs (e.g. failure).

        :return:
        """
        # print("~~~~~~~~~~~")
        # print("Timestep:", self._timesteps)
        n = self._agings.steps_till_next_failure(self._components.alive_components,
                                                 self._thermals.temps,
                                                 self._timesteps)

        # print("Doing", n, "steps!")
        # print("AGINGS BEFORE:")
        # print(self._agings.cur_agings)
        self._agings.do_n_steps(n, self._components.alive_components, self._thermals.temps)
        # print("AGINGS AFTER:")
        # print(self._agings.cur_agings)

        # print("COMPONENTS BEFORE:")
        # print(self._components.workload)
        system_ok = self._components.do_n_steps(n, self._agings.cur_agings)
        # print("COMPONENTS AFTER:")
        # print(self._components.workload)

        # print("THERMALS BEFORE:")
        # print(self._thermals.temps)
        self._thermals.do_n_steps(n, self._components.workload)
        # print("THERMALS AFTER:")
        # print(self._thermals.temps)

        # print("AGINGS ADJUSTMENT BEFORE:")
        # print(self._agings._lambdas)
        self._agings.resample_workload_changes(self._components.workload, self._thermals.temps)
        # print("AGINGS ADJUSTMENT AFTER:")
        # print(self._agings._lambdas)

        self._timesteps += n

        return system_ok

    def print_status(self, timestep):
        """ Print the current system values based on each simulation iteration.

        :param timestep - The current amount of timesteps that the simulator has processed.
        :return: None
        """
        print(timestep)

    def log_timestep(self, timestep):
        """ Write information about the current timestep to a file.

        :param filename_out: file to write to (/out/<filename_out>)
        :return: None
        """
        root_dir = os.path.dirname(os.path.abspath(__file__))
        f = open(root_dir + "/../../out/" + filename_out, "a+")
        f.write("%d %s %f\n" % (timestep, np.sum(self._components.capacities), np.average(self._thermals.temps)))
        f.close()
