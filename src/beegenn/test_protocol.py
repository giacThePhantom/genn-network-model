from .first_protocol import FirstProtocol

class TestFirstProtocol(FirstProtocol):
    def events_generation(self, _):
        """Creates the event for the protocol and saves them in a private field
        Parameters
        ----------
        num_concentration_increases : int
            The number of times the concentration is increased by a dilution factor
        """
        res = []
        t = self.resting_duration
        for (i, odor) in enumerate(self.odors):
            if i >= 3:
                break
            for c_exp in range(15, 18):
                res.append(self._event_generation(t, odor, c_exp))
                t = res[-1]['t_end'] + self.resting_duration
        self.events = res

