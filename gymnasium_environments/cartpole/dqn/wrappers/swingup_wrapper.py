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

        # unwrap the TimeLimit wrapper into CartPoleEnv:
        core = self.env.unwrapped

        # store threshold for termination condition:
        self.x_threshold = core.x_threshold

        # need to change the observation space to accomodate downwards start:
        low  = np.array([-self.x_threshold, -np.inf, -np.pi, -np.inf], dtype=np.float32)
        high = np.array([ self.x_threshold,  np.inf,  np.pi,  np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # redefine the reset function:
    def reset(self, *, seed = None, options = None):
        # reset the TimeLimit wrapped env:
        obs, info = self.env.reset(seed = seed, options = options)

        # unwrap the TimeLimit wrapper into CartPoleEnv:
        core = self.env.unwrapped

        # set the start position to pi:
        state = np.array([0.0, 0.0, np.pi, 0.0], dtype = np.float32)

        # add noise to the initial state:
        state += core.np_random.normal(0, 0.1, size = 4)

        # set the state:
        core.state = tuple(state)

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

        # unpack state:
        x, x_dot, theta, theta_dot = core.state

        # reward_theta is 1 when theta is 0 or 2pi, 0 if between 90 and 270:
        reward_theta = max(0, np.cos(theta))

        # reward_x is 0 when cart is at the edge of the screen, 1 when it's in the center:
        reward_x = np.cos((x / core.x_threshold) * (np.pi / 2.0))

        # reward between [0, 1]:
        reward = reward_theta * reward_x

        # override the termination:
        terminated = bool(abs(x) > core.x_threshold)

        # ensure datatype:
        obs = np.array(obs, dtype = np.float32)

        # return:
        return obs, reward, terminated, truncated, info