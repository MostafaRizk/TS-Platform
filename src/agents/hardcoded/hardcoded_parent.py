import re
from collections import deque
import numpy as np
from agents.agent import Agent


class HardcodedAgent(Agent):
    def __init__(self):
        self.action_index = {"FORWARD": 0, "BACKWARD": 1, "LEFT": 2, "RIGHT": 3, "PICKUP": 4, "DROP": 5}
        self.area_from_bits = {"1000": "NEST", "0100": "CACHE", "0010": "SLOPE", "0001": "SOURCE"}
        self.obstacle_from_bits = {"1": "OBSTACLE", "0": "BLANK"}
        self.sensor_range = 1
        self.agent_position = (1, 1)  # Center position of sensor_map. Assumes sensor range is 1
        self.sensor_map = []
        self.current_zone = None
        self.has_resource = None
        self.last_action = "FORWARD"

    def act(self, observation):
        # Break down observations
        self.sensor_map = self.get_sensor_map(observation)

        self.current_zone = self.area_from_bits[re.sub('[ ,\[\]]', '', str(observation[-5:-1]))]  # Get 4-bit vector representing area, remove brackets and commas, use it as a key for the dictionary of areas

        if observation[-1] == 0:
            self.has_resource = False
        elif observation[-1] == 1:
            self.has_resource = True

        action = None

    def get_sensor_map(self, observation):
        unrefined_map = re.sub('[ ,\[\]]', '', str(observation[0:9]))
        row_length = 3
        col_length = 3

        refined_map = [[None for j in range(col_length)] for i in range(row_length)]

        for i in range(row_length):
            for j in range(col_length):
                refined_map[i][j] = self.obstacle_from_bits[unrefined_map[i*row_length + j]]

        return refined_map
