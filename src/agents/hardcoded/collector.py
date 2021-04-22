import re
import math

from agents.hardcoded.hardcoded_parent import HardcodedAgent


class HardcodedCollectorAgent(HardcodedAgent):
    def __init__(self):
        super().__init__()

    def act(self, observation):
        """
        Use observations to enact a Collector policy (i.e. Get resource from cache and bring it back to nest)
        :param observation:
        :return:
        """

        super().act(observation)

        if not self.has_resource:

            if self.current_zone == "NEST":
                action = self.action_index["FORWARD"]

            elif self.current_zone == "CACHE":
                # If an obstacle is detected but there are no walls
                if "OBSTACLE" in self.sensor_map[0]+self.sensor_map[1]+self.sensor_map[2] and \
                        not (self.sensor_map[0][0] == self.sensor_map[1][0] == self.sensor_map[2][0] == "OBSTACLE") and \
                        not (self.sensor_map[0][2] == self.sensor_map[1][2] == self.sensor_map[2][2] == "OBSTACLE"):

                    action = self.action_index["PICKUP"]

                # If left wall
                elif self.sensor_map[0][0] == self.sensor_map[1][0] == self.sensor_map[2][0] == "OBSTACLE":
                    action = self.action_index["RIGHT"]

                # If right wall
                elif self.sensor_map[0][2] == self.sensor_map[1][2] == self.sensor_map[2][2] == "OBSTACLE":
                    action = self.action_index["LEFT"]

                # If no wall
                else:
                    if self.last_action == self.action_index["RIGHT"]:
                        action = self.action_index["RIGHT"]

                    else:
                        action = self.action_index["LEFT"]

            else:
                action = self.action_index["BACKWARD"]

        elif self.has_resource:

            if self.current_zone != "NEST":
                action = self.action_index["BACKWARD"]

            elif self.current_zone == "NEST":
                action = self.action_index["DROP"]
                #self.has_resource = False

        self.last_action = action

        return action

