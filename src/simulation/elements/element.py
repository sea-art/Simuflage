from abc import ABC, abstractmethod


class SimulatorElement(ABC):
    @abstractmethod
    def step(self, *args):
        pass

    @abstractmethod
    def do_n_steps(self, *args):
        pass
