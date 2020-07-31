import re
import math
import numpy as np

from agents.hardcoded.hardcoded_parent import HardcodedAgent


class HardcodedCollectorAgent(HardcodedAgent):
    def __init__(self, seed):
        super().__init__()
        self.random_state = np.random.RandomState(seed)

    def act(self, observation):
        """
        Use observations to enact a Collector policy (i.e. Get resource from cache and bring it back to nest)
        :param observation:
        :return:
        """

        super().act(observation)

        # Begin behaviour loop
        if not self.has_resource:
            current_tile_contents = self.sensor_map[self.robot_position[1]][self.robot_position[0]]

            # If resource is on the current tile, pick it up
            if current_tile_contents == "RESOURCE":
                action = self.action_index["PICKUP"]

            # Otherwise, find a resource (while avoiding obstacles)
            else:
                action = self.find_resource()

        elif self.has_resource:
            # If on the nest drop the resource
            if self.current_zone == "NEST":
                action = self.action_index["DROP"]

            # Otherwise look for the nest (while avoiding obstacles
            else:
                # Look for the nest
                action = self.action_index["BACKWARD"]

        if self.is_stuck():
            action = self.random_state.randint(low=0, high=len(self.action_index))

        self.memory.append((observation, action))
        return action

    def find_resource(self):
        """
        Do actions that help the robot find the resource
        :return:
        """
        # Move to the cache
        if self.current_zone == "NEST":
            return self.action_index["FORWARD"]

        elif self.current_zone == "SLOPE" or self.current_zone == "SOURCE":
            return self.action_index["BACKWARD"]

        # If on the cache, move towards the edge, move right or left if you're at the edge
        else:
            if "RESOURCE" in self.sensor_map[0]:
                return self.action_index["FORWARD"]
            elif "RESOURCE" in [self.sensor_map[1][0], self.sensor_map[2][0]]:
                return self.action_index["LEFT"]
            elif "RESOURCE" in [self.sensor_map[1][2], self.sensor_map[2][2]]:
                return self.action_index["RIGHT"]
            elif self.sensor_map[1][2] == "WALL":
                return self.action_index["BACKWARD"]
            else:
                return self.action_index["RIGHT"]



