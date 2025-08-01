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
        inf = np.finfo(np.float32).max
        low  = np.array([-x_thresh, -inf, -np.pi, -inf], dtype=np.float32)
        high = np.array([ x_thresh,  inf,  np.pi,  inf], dtype=np.float32)
        self.observation_space = spaces.Box(low, high, dtype=np.float32)

    # redefine the reset function:
    def reset(self, *, seed = None, options = None):
        # reset the TimeLimit wrapped env:
        obs, info = self.env.reset(seed = seed, options = options)

        # unwrap the TimeLimit wrapper into CartPoleEnv:
        core = self.env.unwrapped
        
        if core.np_random.uniform() < 0.25:
            # ocassionally start the cart upright:
            x, x_dot, theta, theta_dot = np.array([0, 0, 0, 0]) + core.np_random.normal(0, 0.1, size = 4)
        else:
            # force pendulum downwards:
            core = self.env.unwrapped 
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
        x, x_dot, theta, theta_dot = core.state

        # define the new reward to encourage bounding theta between -0.1 and 0.1 rad:
        reward = 1.0 if abs(theta) < 0.1 else 0.0

        # override the termination:
        terminated = bool(abs(x) > core.x_threshold)

        # ensure datatype:
        obs = np.array(obs, dtype = np.float32)

        # return:
        return obs, reward, terminated, truncated, info
        