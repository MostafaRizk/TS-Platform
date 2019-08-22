"""
Code modified from OpenAI's Mountain robot example
"""

import gym
from gym import error, spaces, utils
from gym.utils import seeding

import math
import numpy as np
import copy

class TSEnv(gym.Env):
    metadata = {
      'render.modes': ['human', 'rgb_array'],
      'video.frames_per_second': 30
    }

    def __init__(self, goal_velocity = 0):
        '''
        Initialise position, speed, gravity and velocity. Also defines action space and observation space. Creates seed?
        :param goal_velocity:
        '''
        self.robot_width=40
        self.robot_height=20
        self.res_width=20
        self.res_height=30

        self.min_position = 0
        self.max_position = 10
        self.max_speed = 0.07
        self.sensor_range = 1

        self.gravity = 0.0025

        self.cache_start = 1
        self.slope_start = 2.5
        self.slope_end = 7.5
        self.slope_angle = 10

        #self.position = np.random.uniform(low=self.min_position, high=self.cache_start)
        self.position = np.random.uniform(low=2, high=3)
        #self.res_position = np.random.uniform(low=self.slope_end, high=self.max_position)
        self.res_position = np.random.uniform(low=3, high=4)

        self.behaviour_list = [self.phototaxis_step, self.antiphototaxis_step, self.random_walk_step]

        self.viewer = None

        #TODO: Modify to have relevant actions and observations
        self.action_space = spaces.Discrete(4) #0-Phototaxis, 1-Antiphototaxis, 2-Random walk, 3-Flip want
        self.observation_space = spaces.Discrete(4) #position, want_resource, has_resource, behaviour

        self.seed()

        #TODO: Initialise neural network

    def seed(self, seed=None):
        '''
        Creates random seed and returns it
        :param seed:
        :return:
        '''
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        '''
        Logs state, adjusts velocity according to action taken, adjusts position according to velocity and records new state.
        :param action:
        :return: The new state, the reward and whether or not the goal has been achieved
        '''

        #Returns an error if the action is invalid
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))

        area, want_resource, has_resource, behaviour = self.state
        '''
        velocity += (action-1)*self.force + math.cos(3*position)*(-self.gravity)
        velocity = np.clip(velocity, -self.max_speed, self.max_speed)
        position += velocity
        position = np.clip(position, self.min_position, self.max_position)

        if (position==self.min_position and velocity<0):
            velocity = 0
        '''

        #if action is 0,1 or 2, do a new step. if it's 3 or 4, continue current behaviour and do other action
        if action <= 2:
            behaviour = action
        else:
            want_resource = not want_resource

        self.behaviour_list[behaviour]()
        self.position = np.clip(self.position, self.min_position, self.max_position)
        area = self.get_robot_location(self.position)

        if self.res_position - self.sensor_range < self.position < self.res_position + self.sensor_range:
            if want_resource and not has_resource:
                self.pickup_step()
                has_resource = True

        if has_resource and want_resource:
            #print("Has resource")
            self.res_position = copy.deepcopy(self.position)
        else:
            #print("Does not have resource")
            if self.slope_start <= self.res_position < self.slope_end:
                self.slide_cylinder()

        #TODO: Plug observations into NN

        #done = bool(position >= self.goal_position and velocity >= self.goal_velocity)
        done = self.res_position < self.cache_start and not self.state[2] #Done if resource is at the nest and not in the robot's possession
        reward = -1.0

        self.state = (area, want_resource, has_resource, behaviour)
        return np.array(self.state), reward, done, {}

    def reset(self):
        '''
        Sets a new random location between 0 and 1 and sets velocity to 0
        :return:
        '''
        self.position = np.random.uniform(low=2, high=3)
        self.res_position = np.random.uniform(low=3, high=4)
        self.state = np.array([self.get_robot_location(self.position), #Location
                               self.np_random.randint(low=0, high=2),  #Object want_resource (0- WANT_OBJECT=False, 1- WANT_OBJECT=True)
                               0,  #Object has_resource (0- HAS_OBJECT=False, 1- HAS_OBJECT=True)
                               self.np_random.randint(low=0, high=3)]) #Current behaviour (0-Phototaxis, 1-Antiphototaxis, 2-Random walk)
        return np.array(self.state)

    def height_map(self, x):
        '''
        Maps an x coordinate on the track to the height of the track at that point
        :param x:
        :return:
        '''
        if x < self.slope_start:
            return self.slope_start * math.tan(math.radians(self.slope_angle))
        elif x < self.slope_end:
            return x * math.tan(math.radians(self.slope_angle))
        else:
            return self.slope_end * math.tan(math.radians(self.slope_angle))

    def _height(self, xs):
        '''
        Returns height according to the function representing the track
        :param xs:
        :return:
        '''
        if not isinstance(xs,np.ndarray):
            return self.height_map(xs)
        else:
            ys = np.copy(xs)
            for i in range(len(xs)):
                ys[i] = self.height_map(xs[i])
            return ys

    def render(self, mode='human'):
        '''
        Renders the environment and agent(s)
        :param mode:
        :return:
        '''
        screen_width = 600
        screen_height = 400

        world_width = self.max_position - self.min_position
        scale = screen_width/world_width

        if self.viewer is None:
            #Define the geometry of the track
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            xs = np.linspace(self.min_position, self.max_position, 100)
            ys = self._height(xs)
            xys = list(zip((xs-self.min_position)*scale, ys*scale))

            self.track = rendering.make_polyline(xys)
            self.track.set_linewidth(4)
            self.viewer.add_geom(self.track)

            robot_clearance = 10
            res_clearance = 0

            #Create robot
            l,r,t,b = -self.robot_width/2, \
                      self.robot_width/2, \
                      self.robot_height, \
                      0
            robot = rendering.FilledPolygon([(l,b), (l,t), (r,t), (r,b)])
            robot.set_color(0, 0, 1)
            robot.add_attr(rendering.Transform(translation=(0, robot_clearance)))
            self.robottrans = rendering.Transform()
            robot.add_attr(self.robottrans)
            self.viewer.add_geom(robot)
            frontwheel = rendering.make_circle(self.robot_height/2.5)
            frontwheel.set_color(.5, .5, .5)
            frontwheel.add_attr(rendering.Transform(translation=(self.robot_width/4,robot_clearance)))
            frontwheel.add_attr(self.robottrans)
            self.viewer.add_geom(frontwheel)
            backwheel = rendering.make_circle(self.robot_height/2.5)
            backwheel.add_attr(rendering.Transform(translation=(-self.robot_width/4,robot_clearance)))
            backwheel.add_attr(self.robottrans)
            backwheel.set_color(.5, .5, .5)
            self.viewer.add_geom(backwheel)

            #Create resource
            '''
            l, r, t, b = (self.res_position-self.min_position)*scale -self.res_width / 2, \
                         (self.res_position-self.min_position)*scale + self.res_width / 2, \
                         self._height(self.res_position)*scale + self.res_height, \
                         self._height(self.res_position)*scale
            '''
            l, r, t, b = -self.res_width / 2, \
                         self.res_width / 2, \
                         self.res_height, \
                         0
            resource = rendering.FilledPolygon([(l, b), (l, t), (r, t), (r, b)])
            resource.add_attr(rendering.Transform(translation=(0, res_clearance)))
            self.restrans = rendering.Transform()
            resource.add_attr(self.restrans)
            self.viewer.add_geom(resource)
            resource.set_color(0, 1, 0)

        #Set position of robot
        robot_pos = self.position
        self.robottrans.set_translation((robot_pos-self.min_position)*scale, self._height(robot_pos)*scale)
        #self.robottrans.set_translation(robot_pos * scale, self._height(robot_pos) * scale)

        #Set position of resource
        res_pos = self.res_position
        self.restrans.set_translation((res_pos - self.min_position) * scale, self._height(res_pos) * scale)
        #self.restrans.set_translation(res_pos, self._height(res_pos) )
        #self.restrans.set_rotation(math.cos(3 * pos))

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def get_robot_location(self, position):
        '''
        Calculates an integer representing which area the robot is on depending on its position
        :return:
        '''
        #(0 - ON_SOURCE, 1 - ON_CACHE, 2 - ON_SLOPE, 3 - ON_NEST)
        if self.min_position <= position < self.cache_start:
            return 0
        if self.cache_start <= position < self.slope_start:
            return 1
        if self.slope_start <= position < self.slope_end:
            return 2
        else:
            return 3

    def phototaxis_step(self):
        self.position += 1*self.max_speed

    def antiphototaxis_step(self):
        self.position -= 1*self.max_speed

    def random_walk_step(self):
        choice = np.random.randint(low=-1, high=2)
        self.position += choice*self.max_speed

    def pickup_step(self):
        #print("Picking up resource")
        self.res_position = copy.deepcopy(self.position)

    def slide_cylinder(self):
        self.res_position -= 2

    def get_keys_to_action(self):
        return {():1,(276,):0,(275,):2,(275,276):1} #control with left and right arrow keys

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
