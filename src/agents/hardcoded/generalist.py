import re
import math

from agents.hardcoded.hardcoded_parent import HardcodedAgent


class HardcodedGeneralistAgent(HardcodedAgent):
    def __init__(self):
        super().__init__()

    def act(self, observation):
        """
        Use observations to enact a Generalist policy (i.e. Get resource and bring it back to nest w/out collaborating
        with other robots)
        :param observation:
        :return:
        """

        super().act(observation)

        if not self.has_resource:
            if self.current_zone == "SOURCE":
                action = self.action_index["PICKUP"]
            else:
                action = self.action_index["FORWARD"]

        else:
            if self.current_zone == "NEST":
                action = self.action_index["DROP"]
            else:
                action = self.action_index["BACKWARD"]

        self.last_action = action

        return action

