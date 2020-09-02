import re
import math

from agents.hardcoded.hardcoded_parent import HardcodedAgent


class HardcodedDropperAgent(HardcodedAgent):
    def __init__(self):
        super().__init__()

    def act(self, observation):
        """
        Use observations to enact a Dropper policy (i.e. Get resource from source and drop it)
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
            if self.current_zone == "SLOPE":
                action = self.action_index["DROP"]
            else:
                action = self.action_index["BACKWARD"]

        self.last_action = action

        return action

