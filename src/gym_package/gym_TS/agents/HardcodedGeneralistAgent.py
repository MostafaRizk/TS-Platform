import re
import math

from gym_package.gym_TS.agents.HardcodedAgent import HardcodedAgent


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
            if self.has_resource:
                action = self.action_index["RIGHT"]
            else:
                action = self.action_index["LEFT"]

        self.memory.append((observation, action))
        return action

    def find_resource(self):
        """
        Do actions that help the robot find the resource
        :return:
        """
        # If not on the source, move forward
        if self.current_zone != "SOURCE":
            return self.action_index["FORWARD"]

        # If on the source, move towards the edge, move right or left if you're at the edge
        else:
            if self.sensor_map[0] == ["WALL", "WALL", "WALL"]:
                if self.sensor_map[1][2] == "BLANK" or self.sensor_map[1][2] == "RESOURCE":
                    return self.action_index["RIGHT"]
                elif self.sensor_map[1][0] == "BLANK" or self.sensor_map[1][0] == "RESOURCE":
                    return self.action_index["LEFT"]
                else:
                    return self.action_index["BACKWARD"]
            else:
                return self.action_index["FORWARD"]

