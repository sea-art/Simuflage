#!/usr/bin/env python

""" Abstract class for elements of the component."""

from abc import ABC, abstractmethod

__licence__ = "GPL-3.0-or-later"
__copyright__ = "Copyright 2020 Siard Keulen"


class SimulatorElement(ABC):
    @abstractmethod
    def step(self, *args):
        pass

    @abstractmethod
    def do_n_steps(self, *args):
        pass

    @abstractmethod
    def reset(self, *args):
        pass
