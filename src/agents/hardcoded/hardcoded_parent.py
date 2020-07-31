import re
from collections import deque
import numpy as np
from agents.agent import Agent


class HardcodedAgent(Agent):
    def __init__(self):
        self.action_index = {"FORWARD": 0, "BACKWARD": 1, "LEFT": 2, "RIGHT": 3, "PICKUP": 4, "DROP": 5}
        self.area_from_bits = {"1000": "NEST", "0100": "CACHE", "0010": "SLOPE", "0001": "SOURCE"}
        self.obstacle_from_bits = {"1000": "BLANK", "0100": "AGENT", "0010": "RESOURCE", "0001": "WALL"}
        self.sensor_range = 1
        self.robot_position = (1, 1)  # Center position of sensor_map. Assumes sensor range is 1
        self.sensor_map = []
        self.current_zone = None
        self.has_resource = None
        self.memory_length = 4
        self.memory = deque(maxlen=self.memory_length)

    def act(self, observation):
        # Break down observations
        self.sensor_map = self.get_sensor_map(observation)

        self.current_zone = self.area_from_bits[re.sub('[ ,\[\]]', '', str(observation[
                                                                           -5:-1]))]  # Get 4-bit vector representing area, remove brackets and commas, use it as a key for the dictionary of areas
        self.has_resource = bool(observation[-1])

        action = None

    def get_sensor_map(self, observation):
        unrefined_map = [[observation[0:4], observation[4:8], observation[8:12]],
                         [observation[12:16], observation[16:20], observation[20:24]],
                         [observation[24:28], observation[28:32], observation[32:36]]]

        refined_map = []

        for y in range(len(unrefined_map)):
            refined_row = []

            for x in range(len(unrefined_map[y])):
                refined_row += [self.obstacle_from_bits[re.sub('[ ,\[\]]', '', str(unrefined_map[y][x]))]]

            refined_map += [refined_row]

        return refined_map

    def is_stuck(self):
        if len(self.memory) < self.memory_length:
            return False

        all_the_same = True

        for i in range(len(self.memory)-1):
            if not self.memory_is_equal(self.memory[i], self.memory[i+1]):
                all_the_same = False
                break

        if all_the_same:
            return True

        alternates_are_the_same = True

        for i in range(len(self.memory)-2):
            if not self.memory_is_equal(self.memory[i], self.memory[i+2]):
                alternates_are_the_same = False
                break

        if alternates_are_the_same:
            return True

        return False

    def memory_is_equal(self, memory_1, memory_2):
        if np.array_equal(memory_1[0], memory_2[0]) and memory_1[1] == memory_2[1]:
            return True
        else:
            return False
