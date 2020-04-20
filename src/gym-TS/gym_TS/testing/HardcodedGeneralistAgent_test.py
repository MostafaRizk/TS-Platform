import unittest

from gym_TS.agents.HardcodedGeneralistAgent import HardcodedGeneralistAgent


class TestHardcodedGeneralist(unittest.TestCase):
    def test_act(self):
        agent = HardcodedGeneralistAgent()
        contents = {"Blank": [1, 0, 0, 0], "Robot": [0, 1, 0, 0], "Resource": [0, 0, 1, 0], "Wall": [0, 0, 0, 1]}
        areas = {"Nest": [1, 0, 0, 0], "Cache": [0, 1, 0, 0], "Slope": [0, 0, 1, 0], "Source": [0, 0, 0, 1]}
        object = {"Has": [1], "Not has": [0]}

