class Application:
    """ Application that can be mapped to a component."""

    def __init__(self, power):
        """ Initialize an application that can be mapped to a component.

        :param power: The amount of power required to run this application by a component.
        """
        self._power_req = power

    @property
    def power_req(self):
        return self._power_req
