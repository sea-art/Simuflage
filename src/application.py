class Application:
    """ Application that can be mapped to a component."""

    def __init__(self, power):
        """ Initialize an application that can be mapped to a component.

        :param power: The amount of power required to run this application by a component.
        """
        self.power_req = power
