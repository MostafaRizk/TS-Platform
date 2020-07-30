from agents.hardcoded.hardcoded_parent import HardcodedAgent
from agents.hardcoded.dropper import HardcodedDropperAgent
from agents.hardcoded.collector import HardcodedCollectorAgent
import numpy as np


class HardcodedLazyGeneralistAgent(HardcodedAgent):
    def __init__(self, switch_probability, seed):
        super().__init__()
        self.current_strategy = "dropper"
        self.switch_probability = switch_probability
        self.random_state = np.random.RandomState(seed)

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
            if self.current_zone == "SLOPE" or self.current_zone == "NEST":
                action = self.action_index["DROP"]

            # Otherwise go backward
            else:
                action = self.action_index["BACKWARD"]

        if self.is_stuck():
            self.switch()

        self.memory.append((observation, action))
        return action

    def find_resource(self):
        """
        Do actions that help the robot find the resource
        :return:
        """
        if self.current_zone == "SLOPE":
            random_number = self.random_state.uniform()

            if random_number < self.switch_probability:
                self.switch()

        if self.current_strategy == "dropper":

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

        elif self.current_strategy == "collector":
            # Move to the cache
            if self.current_zone == "NEST":
                return self.action_index["FORWARD"]

            elif self.current_zone == "SLOPE" or self.current_zone == "SOURCE":
                return self.action_index["BACKWARD"]

            # If on the cache, move towards the edge, move right or left if you're at the edge
            else:
                if self.sensor_map[1][0] == "BLANK":
                    return self.action_index["LEFT"]
                elif self.sensor_map[1][0] == "WALL":
                    return self.action_index["BACKWARD"]
                else:
                    return self.action_index["RIGHT"]

    def switch(self):
        if self.current_strategy == "dropper":
            self.current_strategy = "collector"

        elif self.current_strategy == "collector":
            self.current_strategy = "dropper"
