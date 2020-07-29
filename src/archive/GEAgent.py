from gym.utils import seeding

# *:*:IS_WANT_OBJECT,false,16;B_RANDOM_WALK,92#
# *:B_PHOTOTAXIS;B_RANDOM_WALK;*:IS_WANT_OBJECT,true,70;IS_MOTIVATION_1_DEC,false,89;B_RANDOM_WALK,9#
# P_ON_SOURCE,true;*:*:B_RANDOM_WALK,67#
# P_MOTIVATION_2_GT,true;*:*:B_PHOTOTAXIS,2#
# P_HAS_OBJECT,true;P_ON_NEST,False*:*:B_RANDOM_WALK,12;IS_MOTIVATION_1_DEC,true,37;IS_MOTIVATION_2_INC,true,3#
# P_ON_CACHE,true;*:*:IS_DROP_OBJECT,false,60#
# P_HAS_OBJECT,true;*:*:B_ANTI_PHOTOTAXIS,81;IS_MOTIVATION_1_DEC,false,76;B_ANTI_PHOTOTAXIS,81;IS_MOTIVATION_1_DEC,true,7;IS_MOTIVATION_1_DEC,true,11#


class GEAgent:
    def __init__(self, rule_string):
        self.rules = rule_string.split('#').strip()
        self.on_nest = False
        self.on_cache = False
        self.on_slope = False
        self.on_source = False
        self.has_resource = False
        self.nothing_detected = False
        self.robot_detected = False
        self.resource_detected = False
        self.wall_detected = False

    def act(self, observation):
        action = None

        self.parse_observation(observation)

        for rule in self.rules:
            preconditions, actions = rule.split(":")

            if self.check_preconditions(preconditions):
                action = self.do_actions(actions)

    def parse_observation(self, observation):
        if observation[0] == 1:
            self.nothing_detected = True
        elif observation[1] == 1:
            self.robot_detected = True
        elif observation[2] == 1:
            self.resource_detected = True
        elif observation[3] == 1:
            self.wall_detected = True

        if observation[4] == 1:
            self.on_nest = True
        elif observation[5] == 1:
            self.on_cache = True
        elif observation[6] == 1:
            self.on_slope = True
        elif observation[7] == 1:
            self.on_source = True

        if observation[8] == 1:
            self.has_resource = True

    def check_preconditions(self, preconditions):
        for precondition in preconditions.split(";"):
            condition, value = precondition.split(",")
            if value == "true":
                value = True
            elif value == "false:":
                value = False

            if condition == "P_ON_NEST":
                if self.on_nest != value:
                    return False
            elif condition == "P_ON_CACHE":
                if self.on_cache != value:
                    return False
            elif condition == "P_ON_SLOPE":
                if self.on_slope != value:
                    return False
            elif condition == "P_ON_SOURCE":
                if self.on_source != value:
                    return False
            elif condition == "P_NOTHING_DETECTED":
                if self.nothing_detected != value:
                    return False
            elif condition == "P_ROBOT_DETECTED":
                if self.robot_detected != value:
                    return False
            elif condition == "P_RESOURCE_DETECTED":
                if self.resource_detected != value:
                    return False
            elif condition == "P_WALL_DETECTED":
                if self.wall_detected != value:
                    return False
            elif condition == "P_HAS_RESOURCE":
                if self.has_resource != value:
                    return False

        return True

    def do_actions(self, actions):
        pass

    def load(self):
        pass

    def save(self):
        pass
