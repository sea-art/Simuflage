class Application:
    """ Application that can be mapped to a component."""

    def __init__(self, power):
        """ Initialize an application that can be mapped to a component.

        :param power: The amount of power required to run this application by a component.
        """
        assert power >= 0, "Power requirement for applications has to be positive"

        self._power_req = power

    @property
    def power_req(self):
        """ Getter function for the power_req instance variable.

        :return: integer indicating the power requirement for running this application.
        """
        return self._power_req
