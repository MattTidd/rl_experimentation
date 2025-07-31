"""
this wrapper is for gymnasium's cartpole environment. the idea is that it reshapes
the reward and changes the starting state.

"""
# import these packages:
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# define the class for the wrapper:
class SwingUpWrapper(gym.Wrapper):
    # constructor:
    def __init__(self, env):
        # inherit from env:
        super().__init__(env)

        # need to change the observation space to accomodate downwards start:
        # unwrap the TimeLimit wrapper into CartPoleEnv:
        core = self.env.unwrapped

        # set bounds:
        x_thresh = core.x_threshold
        low  = np.array([-x_thresh, -np.inf, -np.pi, -np.inf], dtype=np.float32)
        high = np.array([ x_thresh,  np.inf,  np.pi,  np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # redefine the reset function:
    def reset(self, *, seed = None, options = None):
        # reset the TimeLimit wrapped env:
        obs, info = self.env.reset(seed = seed, options = options)

        # force pendulum downwards on the "raw" CartPoleEnv:
        core = self.env.unwrapped   # unwrap the TimeLimit wrapper into CartPoleEnv
        x, x_dot, theta, theta_dot = np.array([0, 0, np.pi, 0]) + core.np_random.normal(0, 0.1, size = 4)

        # set the state:
        core.state = (x, x_dot, theta, theta_dot)

        # rebuild an observation from that state:
        obs = np.array(core.state, dtype = np.float32)

        # return:
        return obs, info
    
    # redefine the step function:
    def step(self, action):
        # get returns from env.step:
        obs, _, terminated, truncated, info = self.env.step(action)

        # unwrap the TimeLimit wrapper into CartPoleEnv:
        core = self.env.unwrapped

        # unpack true angle from internal state:
        theta = core.state[2]

        # define the new reward = cos(theta) (maximize upright)
        reward = 1 + np.cos(theta)

        # ensure datatype:
        obs = np.array(obs, dtype = np.float32)

        # return:
        return obs, reward, terminated, truncated, info

        