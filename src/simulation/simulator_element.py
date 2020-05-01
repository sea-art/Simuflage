from abc import ABC, abstractmethod


class SimulatorElement(ABC):
    @abstractmethod
    def step(self):
        pass
