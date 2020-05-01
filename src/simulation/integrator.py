from abc import ABC, abstractmethod


class AbsIntegrator(ABC):
    @abstractmethod
    def step(self, *args):
        pass


class Integrator(AbsIntegrator):
    """ This class should be edited when adding new elements or changing simulation functionality.

    """
    def __init__(self, components, thermals, agings):
        self._components = components
        self._thermals = thermals
        self._agings = agings

    def step(self):
        self._thermals.step(self._components.comp_loc_map)
        remap_required = self._agings.step(self._components.alive_components, self._thermals.temps)
        system_ok = self._components.step(self._agings.cur_agings)

        return system_ok
