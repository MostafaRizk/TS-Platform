import re
import math

from agents.HardcodedAgent import HardcodedAgent


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

        # Begin behaviour loop
        if not self.has_resource:
            current_tile_contents = self.sensor_map[self.robot_position[1]][self.robot_position[0]]

            # If resource is on the current tile, pick it up
            if current_tile_contents == "RESOURCE":
                return self.action_index["PICKUP"]

            # Otherwise, find a resource (while avoiding obstacles)
            else:
                return self.find_resource()

        elif self.has_resource:
            # If on the nest drop the resource
            if self.current_zone == "SLOPE":
                return self.action_index["DROP"]

            # Otherwise look for the slope (while avoiding obstacles
            else:
                # Look for the slope
                return self.action_index["BACKWARD"]

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

