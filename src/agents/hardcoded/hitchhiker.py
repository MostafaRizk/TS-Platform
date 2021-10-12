import re
import math

from agents.hardcoded.hardcoded_parent import HardcodedAgent


class HardcodedHitchhikerAgent(HardcodedAgent):
    def __init__(self):
        super().__init__()

    def act(self, observation):
        """
        Use observations to enact a Hitchhiking policy (i.e. Do nothing all the time and hope other agents do the work)
        :param observation:
        :return:
        """

        #super().act(observation)

        action = self.action_index["DROP"]

        self.last_action = action

        return action

