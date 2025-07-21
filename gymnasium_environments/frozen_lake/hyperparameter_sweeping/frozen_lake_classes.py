####################### IMPORTING #######################
import gymnasium as gym
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

####################### CLASSES #######################
# MC-Agent Class:
class GLIE_MC_Agent:
        ####################### INITIALIZATION #######################
        # constructor:
        def __init__(self, 
                     env: gym.Env, 
                     gamma: float, 
                     initial_epsilon: float, 
                     epsilon_decay: float, 
                     final_epsilon: float,
                     es : bool, 
                     rs: bool):
                """
                this is the constructor for the agent. this agent is a monte-carlo agent, meaning that it averages the returns
                for each Q(s,a) at the end of the episode

                env:    a gymnasium environment
                gamma:  a float value indicating the discounting factor
                initial_epsilon:        a float value indicating the starting ε
                epsilon_decay:          a float value indicating the decay rate of ε
                final_epsilon:          a float value indicating the final ε
                es:     a boolean value indicating whether to use exploring starts or not
                rs:     a boolean value indicating whether to use custom rewards or not
                                if true:
                                        goal_value:     +10.0
                                        hole_value:     -1.0
                                else:
                                        goal_value:     +1.0
                                        hole_value:     0.0 (sparsely defined)
                Q:      the estimate of the action-value function q, initialized as zeros over all states and actions
                
                """
                # object parameters:
                self.env = env
                self.gamma = gamma
                self.epsilon = initial_epsilon
                self.epsilon_decay = epsilon_decay
                self.final_epsilon = final_epsilon
                self.es = es
                self.rs = rs

                # reward shaping:
                if self.rs:
                        self.goal_value = 10.0
                        self.hole_value = -1.0
                else:
                        self.goal_value = 1.0
                        self.hole_value = 0.0

                # get the number of states, number of actions:
                nS, nA = env.observation_space.n, env.action_space.n

                # get the terminal spaces of the current map:
                desc = env.unwrapped.desc.astype('U1')
                chars = desc.flatten()
                self.terminal_states = [i for i, c in enumerate(chars) if c in ('H','G')]

                # tabular Q-values, and counter N(s,a):
                self.Q = np.zeros((nS, nA))
                self.visits = np.zeros((nS, nA), dtype = int)         # how many times I have been to a state, and taken an action         

                # return to the user the metrics about the environment:
                print(f"Action Space is: {env.action_space}")
                print(f"Observation Space is: {env.observation_space}\n")
        
        ####################### TRAINING #######################
        # function to perform ε-greedy probability assignment:
        def get_action_probs(self, Q):
                """ 
                this function does the ε-greedy probability assignment for the actions available in a given state

                Q:         a np.ndarray corresponding to the action-values of the actions available in a given state
                returns:   probability of selecting each action
                
                """
                # get the number of available actions:
                m = len(Q)

                # assign each action a base probability of ε/m
                p = np.ones(m)*(self.epsilon/m)

                # find the index of the best Q value
                best = np.argmax(Q)

                # give that one more probability by an amount equal to (1 - ε):
                p[best] += 1.0 - self.epsilon

                # this way the "best" action has a probability of ε/m + (1-ε), meaning it will be chosen more often
                # whereas the others have a probability of ε/m, so there is a probability that exploratory actions will be selected

                # return the probability of selecting each action:
                return p
        
        # ε-greedy policy function:
        def policy(self, state):
                """ 
                this is the ε-greedy policy itself, where it chooses an action based on the ε-greedy probabilities of each action

                state:   an int representing the current state
                returns: a randomly selected action

                """
                probs = self.get_action_probs(self.Q[state])    # for a given state, or row in Q
                return np.random.choice(len(probs), p = probs)  # pick an action from the probabilities of each action
        
        # epsilon decay function:
        def decay_epsilon(self):
                """
                this function is responsible for decaying the value of ε, thereby 
                reducing the exploration rate each episode

                """
                self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

        # episode generation function:
        def generate_episode(self):
                """ 
                this function is used to generate and run through episodes

                returns: a list of (obs, a, r) tuples

                """
                episode = []    # empty list for returns

                # exploring starts:
                if self.es:
                        non_terminals = [s for s in range(self.env.observation_space.n) if s not in self.terminal_states]
                        starting_state = np.random.choice(non_terminals)

                        # force env into starting state:
                        _, _ = self.env.reset()
                        self.env.unwrapped.s = starting_state 
                        obs = starting_state
                else:
                        obs, _ = self.env.reset()

                # flag for when to stop episode
                done = False

                while not done:
                        a = self.policy(obs)    # select an action based on the current state

                        next_obs, r, term, trunc, _ = self.env.step(a)    # take the action

                        # custom reward shaping:
                        if self.rs:
                                if term and r == 0:
                                        r = self.hole_value  # fell in hole
                                elif term and r == 1:
                                        r = self.goal_value  # reached goal

                        episode.append((obs, a, r))     # trajectories are given by {S_1, A_1, R_2, ... , S_T} 
                        obs = next_obs          # advance the state
                        done = term or trunc    # set done to True if term (terminal state) or trunc (environment cut episode short)

                # return episode information for GPI
                return episode
        
        def update_Q(self, episode):
                """ 
                this function updates the Q estimation using incremental every-visit MC at the end of an episode

                episode: the (s, a, r) list

                """
                g = 0   # initial return value
                for (s, a, r) in reversed(episode):
                        g = r + self.gamma * g  # get the return for that state
                        self.visits[s, a] += 1  # increment visits
                        n = self.visits[s, a]   # visit counter is used in the update rule

                        # update Q(s, a) based on update rule:
                        self.Q[s, a] += (g - self.Q[s, a]) / n
        
        # actual policy iteration:
        def GPI(self, num_episodes):
                """ 
                this function performs the generalized policy iteration, using GLIE evaluation and ε-greedy policy improvement

                num_episode: number of episodes to play out
                returns:     the updated Q values
                
                """
                for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                        # 1) play out an entire episode:
                        episode = self.generate_episode()

                        # 2) perform incremental, every-visit MC after the episode to approximate Q(s,a):
                        self.update_Q(episode)

                        # 3) GLIE uses a decaying ε schedule:
                        self.decay_epsilon()

                return self.Q
        
        ####################### EVALUATION #######################
        # average return per episode:
        def average_return(self, num_episodes):
                """ 
                this function computes the average return per episode for a given amount of episodes

                agent:          the agent that has been trained
                num_episode:    number of episodes to play out
                returns:        the average return per episode
                
                """
                # initialize the total return received over the evaluation:
                total_return = 0

                # for every episode:
                for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                        obs, _ = self.env.reset()      # must reset before an episode
                        done = False                    # flag is set to False initially
                        episode_return = 0              # reset return for the episode

                        # while False:
                        while not done:
                                a = np.argmax(self.Q[obs])                     # pick best action from policy
                                obs, r, term, trunc, _ = self.env.step(a)      # step that action
                                episode_return += r     # increment the episode return by that return
                                done = term or trunc    # set to True if term or trunc
                        
                        total_return += episode_return  # increment total return by episode return
                
                return round(total_return / num_episodes, 3)      # average return accross all episodes
        
        # success rate:
        def success_rate(self, num_episodes):
                """ 
                this function computes the success rate for a given amount of episodes

                agent:          the agent that has been trained
                num_episode:    number of episodes to play out
                returns:        the success rate for that stretch of episodes
                
                """
                # initialize number of successes:
                success = 0

                # for every episode:
                for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                        obs, _ = self.env.reset()      # must reset before an episode
                        done = False                    # flag is set to False initially

                        # while False:
                        while not done:
                                a = np.argmax(self.Q[obs])                     # pick best action from policy
                                obs, r, term, trunc, _ = self.env.step(a)      # step that action
                                done = term or trunc    # set to True if term or trunc

                        # if at the goal pose
                        if r == 1.0:
                                success += 1    # increment the success counter

                return round((success / num_episodes) * 100, 3)   # return success rate
        
        # average episode length:
        def average_length(self, num_episodes):
                """ 
                this function computes the average episode length for a given amount of episodes

                agent:          the agent that has been trained
                num_episodes:   number of episodes to play out
                returns:        the average episode length for that stretch of episodes
                
                """
                # initialize the total number of steps over the evaluation:
                total_steps = 0
                
                # for every episode:
                for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                        obs, _ = self.env.reset()      # must reset before an episode
                        done = False                    # flag is set to False initially
                        episode_steps = 0               # reset steps for the episode

                        # while False:
                        while not done:
                                a = np.argmax(self.Q[obs])                     # pick best action from policy
                                obs, _, term, trunc, _ = self.env.step(a)      # step that action
                                episode_steps += 1                              # increment episode steps

                                done = term or trunc    # set to True if term or trunc

                        total_steps += episode_steps    # increment total steps by steps taken in episode
                
                # return the average steps per episode to the user:
                return round(total_steps / num_episodes, 3)

# SARSA(0)-Agent Class:
class SARSA_0_Agent:
    ####################### INITIALIZATION #######################
    # constructor:
    def __init__(self, 
                env: gym.Env, 
                gamma: float, 
                alpha: float,
                initial_epsilon: float,
                epsilon_decay: float, 
                final_epsilon: float,
                es: bool, 
                rs: bool):
        """ 
        this is the constructor for the agent. this agent is a TD-based agent, implementing SARSA, which means that the 
        policy is evaluated and improved every time-step

        env:    a gymnasium environment
        gamma:  a float value indicating the discount factor
        alpha:  a float value indicating the learning rate
        initial_epsilon:        a float value indicating the starting ε
        epsilon_decay:          a float value indicating the decay rate of ε
        final_epsilon:          a float value indicating the final ε
        es:     a boolean value indicating whether to use exploring starts or not
        rs:     a boolean value indicating whether to use reward shaping or not
                    if true:
                        goal_value: +10.0
                        hole_value: -1.0
                    else:
                        goal_value: +1.0
                        hole_value: 0.0 (sparsely defined)
        Q:      the estimate of the action-value function q, initialized as zeroes over all states and actions

        """
        # object parameters:
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.es = es
        self.rs = rs

        # set the reward shaping:
        if self.rs:
            self.goal_value = 10.0
            self.hole_value = -1.0
        else:
            self.goal_value = 1.0
            self.hole_value = 0.0

        # get the number of states, number of actions:
        self.nS, self.nA = env.observation_space.n, env.action_space.n

        # get the terminal spaces of the current map:
        desc = env.unwrapped.desc.astype("U1")
        chars = desc.flatten()
        self.terminal_states = [i for i, c in enumerate(chars) if c in ("H", "G")]

        # tabular Q-values:
        self.Q = np.zeros((self.nS, self.nA))

        # return to the user the metrics about the environment:
        print(f"Action Space is: {env.action_space}")
        print(f"Observation Space is: {env.observation_space}\n")

    ####################### TRAINING #######################
    # function to perform ε-greedy probability assignment:
    def get_action_probs(self, Q):
        """ 
        this function does the ε-greedy probability assignment for the actions available in a given state

        Q:          a np.ndarray corresponding to the action-values of the actions available in a given state
        returns:    probability of selecting each action

        """
        # get the number of available actions:
        m = len(Q)

        # assign each action a base probability of ε/m:
        p = np.ones(m)*(self.epsilon/m)

        # find the index of the best Q value:
        best = np.argmax(Q)

        # give that one more probability by an amount equal to (1 - ε):
        p[best] += 1.0 - self.epsilon

        # this way the "best" action has a probability of ε/m + (1 - ε), meaning it will be chosen more often
        # whereas the others have a probability of ε/m, so there is a probability that exploratory actions will be selected

        # return the probability of selecting each action:
        return p

    # ε-greedy policy function:
    def policy(self, state):
        """ 
        this is the ε-greedy policy itself, where it chooses an action based on the ε-greedy probabilities of each action

        state:      an int representing the current state
        returns:    a randomly selected action

        """
        probs = self.get_action_probs(self.Q[state])    # for a given state, or row in Q
        return np.random.choice(len(probs), p = probs)  # pick an action from the probabilities of each action

    # epsilon decay function:
    def decay_epsilon(self):
        """
        this function is responsible for decaying the value of ε, thereby 
        reducing the exploration rate each episode

        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # GPI function using SARSA rule and ε-greedy policy:
    def GPI(self, num_episodes):
        """
        this function performs the generalized policy iteration using SARSA as the evaluation modality and
        ε-greedy policy improvement to improve the policy

        num_episodes:   number of desired episodes to train the agent on
        returns:        the updated Q values

        """
        for _ in tqdm(range(num_episodes), colour = '#33FF00', ncols = 100):
            # if exploring starts:
            if self.es:
                non_terminals = [s for s in range(self.env.observation_space.n) if s not in self.terminal_states]
                starting_state = np.random.choice(non_terminals)

                # force env into starting state:
                _, _ = self.env.reset()
                self.env.unwrapped.s = starting_state
                obs = starting_state
            else:
                obs, _ = self.env.reset()
            
            # take an initial ε-greedy action:
            action = self.policy(obs)

            # flag for finishing:
            done = False

            # while False:
            while not done:
                next_obs, r, term, trunc, _ = self.env.step(action)     # take an action, get a new state and a reward
                next_action = self.policy(next_obs)                     # pick next action based on ε-greedy policy

                # if reward shaping:
                if self.rs:
                    if term and r == 0:
                        r = self.hole_value     # fell in a hole
                    elif term and r == 1:
                        r = self.goal_value     # reached goal

                # update Q using SARSA update rule:
                self.Q[obs, action] += self.alpha * (r + self.gamma*self.Q[next_obs, next_action] - self.Q[obs, action])

                # advance state and action indices:
                obs, action = next_obs, next_action

                # check for completion:
                done = term or trunc

            # decay ε:
            self.decay_epsilon()

        return self.Q

    ####################### EVALUATION #######################
    # average return per episode:
    def average_return(self, num_episodes):
            """ 
            this function computes the average return per episode for a given amount of episodes

            agent:          the agent that has been trained
            num_episode:    number of episodes to play out
            returns:        the average return per episode
            
            """
            # initialize the total return received over the evaluation:
            total_return = 0

            # for every episode:
            for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                    obs, _ = self.env.reset()      # must reset before an episode
                    done = False                    # flag is set to False initially
                    episode_return = 0              # reset return for the episode

                    # while False:
                    while not done:
                            a = np.argmax(self.Q[obs])                     # pick best action from policy
                            obs, r, term, trunc, _ = self.env.step(a)      # step that action
                            episode_return += r     # increment the episode return by that return
                            done = term or trunc    # set to True if term or trunc
                    
                    total_return += episode_return  # increment total return by episode return
            
            return round(total_return / num_episodes, 3)      # average return accross all episodes
    
    # success rate:
    def success_rate(self, num_episodes):
            """ 
            this function computes the success rate for a given amount of episodes

            agent:          the agent that has been trained
            num_episode:    number of episodes to play out
            returns:        the success rate for that stretch of episodes
            
            """
            # initialize number of successes:
            success = 0

            # for every episode:
            for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                    obs, _ = self.env.reset()      # must reset before an episode
                    done = False                    # flag is set to False initially

                    # while False:
                    while not done:
                            a = np.argmax(self.Q[obs])                     # pick best action from policy
                            obs, r, term, trunc, _ = self.env.step(a)      # step that action
                            done = term or trunc    # set to True if term or trunc

                    # if at the goal pose
                    if r == 1.0:
                            success += 1    # increment the success counter

            return round((success / num_episodes) * 100, 3)   # return success rate
    
    # average episode length:
    def average_length(self, num_episodes):
        """ 
        this function computes the average episode length for a given amount of episodes

        agent:          the agent that has been trained
        num_episodes:   number of episodes to play out
        returns:        the average episode length for that stretch of episodes
        
        """
        # initialize the total number of steps over the evaluation:
        total_steps = 0
        
        # for every episode:
        for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                obs, _ = self.env.reset()      # must reset before an episode
                done = False                    # flag is set to False initially
                episode_steps = 0               # reset steps for the episode

                # while False:
                while not done:
                        a = np.argmax(self.Q[obs])                     # pick best action from policy
                        obs, _, term, trunc, _ = self.env.step(a)      # step that action
                        episode_steps += 1                              # increment episode steps

                        done = term or trunc    # set to True if term or trunc

                total_steps += episode_steps    # increment total steps by steps taken in episode
        
        # return the average steps per episode to the user:
        return round(total_steps / num_episodes, 3)

# SARSA(λ)-Agent Class:
class SARSA_L_Agent:
    ####################### INITIALIZATION #######################
    # constructor:
    def __init__(self, 
                env: gym.Env, 
                gamma: float, 
                alpha: float, 
                lamb: float, 
                initial_epsilon: float,
                epsilon_decay: float, 
                final_epsilon: float, 
                es: bool, 
                rs: bool):
        """
        this is the constructor for the agent. this agent is a TD-based agent, implementing SARSA(λ), meaning that the policy
        is evaluated and improved every time-step by examining all n-step returns

        env:    a gymnasium environment
        gamma:  a float value indicating the discount factor
        alpha:  a float value indicating the learning rate
        lamb: a float value indicating the trace decay rate, λ
        initial_epsilon:        a float value indicating the starting ε
        epsilon_decay:          a float value indicating the decay rate of ε
        final_epsilon:          a float value indicating the final ε
        es:     a boolean value indicating whether to use exploring starts or not
        rs:     a boolean value indicating whether to use reward shaping or not
                if true:
                    goal_value: +10.0
                    hole_value: -1.0
                else:
                    goal_value: +1.0
                    hole_value: 0.0 (sparsely defined)
        Q:      the estimate of the action-value function q, initialized as zeroes over all states and actions
        E:      the eligibility trace, initialized as zeroes over all states and actions

        """

        # object parameters:
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.lamb = lamb
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.es = es
        self.rs = rs

        # set the reward shaping:
        if self.rs:
            self.goal_value = 10.0
            self.hole_value = -1.0
        else:
            self.goal_value = 1.0
            self.hole_value = 0.0

        # get the number of states, number of actions:
        self.nS, self.nA = env.observation_space.n, env.action_space.n

        # get the terminal spaces of the current map:
        desc = env.unwrapped.desc.astype("U1")
        chars = desc.flatten()
        self.terminal_states = [i for i, c in enumerate(chars) if c in ("H", "G")]

        # tabular Q values:
        self.Q = np.zeros((self.nS, self.nA))

        # return to the user the metrics about the environment:
        print(f"Action Space is: {env.action_space}")
        print(f"Observation Space is: {env.observation_space}\n")

    ####################### TRAINING #######################
    # function to perform ε-greedy probability assignment:
    def get_action_probs(self, Q):
        """ 
        this function does the ε-greedy probability assignment for the actions available in a given state

        Q:          a np.ndarray corresponding to the action-values of the actions available in a given state
        returns:    probability of selecting each action

        """
        # get the number of available actions:
        m = len(Q)

        # assign each action a base probability of ε/m:
        p = np.ones(m)*(self.epsilon/m)

        # find the index of the best Q value:
        best = np.argmax(Q)

        # give that one more probability by an amount equal to (1 - ε):
        p[best] += 1.0 - self.epsilon

        # this way the "best" action has a probability of ε/m + (1-ε), meaning it will be chosen more often
        # whereas the others have a probability of ε/m, so there is a probability that exploratory actions will be selected

        # return the probability of selecting each action:
        return p

    # ε-greedy policy function:
    def policy(self, state):
        """ 
        this is the ε-greedy policy itself, where it chooses an action based on the ε-greedy probabilities of each action

        state:      an int representing the current state
        returns:    a randomly selected action

        """
        probs = self.get_action_probs(self.Q[state])    # for a given state, or row in Q
        return np.random.choice(len(probs), p = probs)  # pick an action from the probabilities of each action

    # epsilon decay function:
    def decay_epsilon(self):
        """
        this function is responsible for decaying the value of ε, thereby 
        reducing the exploration rate each episode

        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # GPI function using backward-view SARSA(λ) update rule and ε-greedy policy:
    def GPI(self, num_episodes):
        """ 
        this function performs the generalized policy iteration using backward-view SARSA(λ) as the evaluation modality and
        ε-greedy policy improvement to improve the policy

        num_episodes:   number of desired episodes to train the agent on
        returns:        the updated Q values

        """
        for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
            # reset the eligibility trace:
            self.E = np.zeros((self.nS, self.nA))

            # if exploring starts:
            if self.es:
                non_terminals = [s for s in range(self.env.observation_space.n) if s not in self.terminal_states]
                starting_state = np.random.choice(non_terminals)

                # force env into starting state:
                _, _ = self.env.reset()
                self.env.unwrapped.s = starting_state
                obs = starting_state
            else:
                obs, _ = self.env.reset()

            # ε-greedily select an action:
            action = self.policy(obs)

            # flag for finishing:
            done = False

            # while False:
            while not done:
                next_obs, r, term, trunc, _ = self.env.step(action)     # take action A, observe R, S'
                next_action = self.policy(next_obs)                     # choose A' from S' using ε-greedy policy Q

                # if reward shaping:
                if self.rs:
                    if term and r == 0:
                        r = self.hole_value     # fell in a hole
                    elif term and r == 1:
                        r = self.goal_value     # reached goal
                    
                # compute delta:
                delta = r + self.gamma*self.Q[next_obs, next_action] - self.Q[obs, action]

                # advance trace:
                self.E[obs, action] += 1

                for s in range(self.nS):
                    for a in range(self.nA):
                        if s not in self.terminal_states:
                            self.Q[s, a] += self.alpha*delta*self.E[s, a]
                        self.E[s, a] = self.gamma*self.lamb*self.E[s, a]
                
                # advance state and action indicies:
                obs, action = next_obs, next_action

                # check for completion:
                done = term or trunc
            
            # decay ε:
            self.decay_epsilon()

        return self.Q

    ####################### EVALUATION #######################
    # average return per episode:
    def average_return(self, num_episodes):
        """ 
        this function computes the average return per episode for a given amount of episodes

        agent:          the agent that has been trained
        num_episode:    number of episodes to play out
        returns:        the average return per episode

        """
        # initialize the total return received over the evaluation:
        total_return = 0

        # for every episode:
        for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
            obs, _ = self.env.reset()      # must reset before an episode
            done = False                    # flag is set to False initially
            episode_return = 0              # reset return for the episode

            # while False:
            while not done:
                a = np.argmax(self.Q[obs])                     # pick best action from policy
                obs, r, term, trunc, _ = self.env.step(a)      # step that action
                episode_return += r     # increment the episode return by that return
                done = term or trunc    # set to True if term or trunc

            total_return += episode_return  # increment total return by episode return

        return round(total_return / num_episodes, 3)      # average return accross all episodes

    # success rate:
    def success_rate(self, num_episodes):
        """ 
        this function computes the success rate for a given amount of episodes

        agent:          the agent that has been trained
        num_episode:    number of episodes to play out
        returns:        the success rate for that stretch of episodes

        """
        # initialize number of successes:
        success = 0

        # for every episode:
        for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
            obs, _ = self.env.reset()      # must reset before an episode
            done = False                    # flag is set to False initially

            # while False:
            while not done:
                a = np.argmax(self.Q[obs])                     # pick best action from policy
                obs, r, term, trunc, _ = self.env.step(a)      # step that action
                done = term or trunc    # set to True if term or trunc

            # if at the goal pose
            if r == 1.0:
                success += 1    # increment the success counter

        return round((success / num_episodes) * 100, 3)   # return success rate

    # average episode length:
    def average_length(self, num_episodes):
        """ 
        this function computes the average episode length for a given amount of episodes

        agent:          the agent that has been trained
        num_episodes:   number of episodes to play out
        returns:        the average episode length for that stretch of episodes

        """
        # initialize the total number of steps over the evaluation:
        total_steps = 0

        # for every episode:
        for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
            obs, _ = self.env.reset()      # must reset before an episode
            done = False                    # flag is set to False initially
            episode_steps = 0               # reset steps for the episode

            # while False:
            while not done:
                a = np.argmax(self.Q[obs])                     # pick best action from policy
                obs, _, term, trunc, _ = self.env.step(a)      # step that action
                episode_steps += 1                              # increment episode steps

                done = term or trunc    # set to True if term or trunc

            total_steps += episode_steps    # increment total steps by steps taken in episode

        # return the average steps per episode to the user:
        return round(total_steps / num_episodes, 3)

# Q-Learning agent class:
class Q_Agent:
    ####################### INITIALIZATION #######################
    # constructor:
    def __init__(self, 
                 env : gym.Env,
                 gamma: float, 
                 alpha: float,
                 initial_epsilon: float,
                 epsilon_decay: float,
                 final_epsilon: float,
                 es: bool,
                 rs: bool):
        """ 
        this is the constructor for the agent. this agent is an off-policy Q-Learning agent, meaning that the learned policy differs
        from the policy that is used to learn that policy. this allows us to learn about an optimal policy while following an exploratory policy

        env:        a gymnasium environment
        gamma:      a float value indicating the discount factor
        alpha:      a float value indicating the learning rate
        initial epsilon:        a float value indicating the starting ε
        epsilon_decay:          a float value indicating the decay rate of ε
        final_epsilon:          a float vlaue indicating the final ε
        es:         a boolean value indicating whether to use exploring starts or not
        rs:         a boolean value indicating whether to use reward shaping or not
                        if true:
                            goal_value: +10.0
                            hole_value: -1.0
                        else:
                            goal_value: +1.0
                            hole_value: 0.0 (sparsely defined)
        Q:          the estimate of the action-value function of q, initialized as zeros over all states and actions
        
        """
        # object parameters:
        self.env = env
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.es = es
        self.rs = rs

        # set the reward shaping:
        if self.rs:
            self.goal_value = 10.0
            self.hole_value = -1.0

        # get the number of states, number of actions:
        self.nS, self.nA = env.observation_space.n, env.action_space.n

        # get the terminal spaces of the current map:
        desc = env.unwrapped.desc.astype("U1")
        chars = desc.flatten()
        self.terminal_states = [i for i, c in enumerate(chars) if c in ("H", "G")]

        # tabular Q-values:
        self.Q = np.zeros((self.nS, self.nA))

        # return to the user the metrics about the environment:
        print(f"Action Space is: {env.action_space}")
        print(f"Observation Space is: {env.observation_space}\n")
    
    ####################### TRAINING #######################
    # function to perform ε-greedy probability assignment:
    def get_action_probs(self, Q):
        """ 
        this function does the ε-greedy probability assignment for the actions available in a given state

        Q:          a np.ndarray corresponding to the action-values of the actions available in a given state
        returns:    probability of selecting each action

        """
        # get the number of available actions:
        m = len(Q)

        # assign each action a base probability of ε/m:
        p = np.ones(m)*(self.epsilon/m)

        # find the index of the best Q value:
        best = np.argmax(Q)

        # give that one more probability by an amount equal to (1 - ε):
        p[best] += 1.0 - self.epsilon

        # this way the "best" action has a probability of ε/m + (1 - ε), meaning it will be chosen more often
        # whereas the others have a probability of ε/m, so there is a probability that exploratory actions will be selected

        # return the probability of selecting each action:
        return p
    
    # ε-greedy policy function:
    def policy(self, state):
        """ 
        this is the ε-greedy policy itself, where it chooses an action based on the ε-greedy probabilities of each action

        state:      an int representing the current state
        returns:    a randomly selected action

        """
        probs = self.get_action_probs(self.Q[state])    # for a given state, or row in Q
        return np.random.choice(len(probs), p = probs)  # pick an action from the probabilities of each action

    # epsilon decay function:
    def decay_epsilon(self):
        """
        this function is responsible for decaying the value of ε, thereby 
        reducing the exploration rate each episode

        """
        self.epsilon = max(self.final_epsilon, self.epsilon - self.epsilon_decay)

    # GPI function based on Q-Learning algorithm:
    def GPI(self, num_episodes):
        """
        this function performs the generalized policy iteration using Q-Learning as the update algorithm.

        num_episodes:     number of desired episodes to train the agent on
        returns:          the updated Q values

        """
        for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
            # if exploring starts:
            if self.es:
                non_terminals = [s for s in range(self.env.observation_space.n) if s not in self.terminal_states]
                starting_state = np.random.choice(non_terminals)

                # force env into starting state:
                _, _ = self.env.reset()
                self.env.unwrapped.s = starting_state
                obs = starting_state
            else:
                obs, _ = self.env.reset()

            # flag for finishing:
            done = False

            # while False:
            while not done:
                # choose A from S using policy derived from Q (e.g., ε-greedy):
                action = self.policy(obs)                      
                
                # take action A, observe R, S':
                next_obs, r, term, trunc, _ = self.env.step(action)

                # if reward shaping:
                if self.rs:
                    if term and r == 0:
                        r = self.hole_value     # fell in a hole
                    elif term and r == 1:
                        r = self.goal_value     # reached goal

                # update Q using Q-Learning update rule:
                self.Q[obs, action] += self.alpha * (r + self.gamma * max(self.Q[next_obs, :]) - self.Q[obs, action])

                # advance state:
                obs = next_obs

                # check for completion:
                done = term or trunc
            
            # decay ε:
            self.decay_epsilon()

        return self.Q

    ####################### EVALUATION #######################
    # average return per episode:
    def average_return(self, num_episodes):
            """ 
            this function computes the average return per episode for a given amount of episodes

            agent:          the agent that has been trained
            num_episode:    number of episodes to play out
            returns:        the average return per episode
            
            """
            # initialize the total return received over the evaluation:
            total_return = 0

            # for every episode:
            for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                    obs, _ = self.env.reset()      # must reset before an episode
                    done = False                    # flag is set to False initially
                    episode_return = 0              # reset return for the episode

                    # while False:
                    while not done:
                            a = np.argmax(self.Q[obs])                     # pick best action from policy
                            obs, r, term, trunc, _ = self.env.step(a)      # step that action
                            episode_return += r     # increment the episode return by that return
                            done = term or trunc    # set to True if term or trunc
                    
                    total_return += episode_return  # increment total return by episode return
            
            return round(total_return / num_episodes, 3)      # average return accross all episodes
    
    # success rate:
    def success_rate(self, num_episodes):
            """ 
            this function computes the success rate for a given amount of episodes

            agent:          the agent that has been trained
            num_episode:    number of episodes to play out
            returns:        the success rate for that stretch of episodes
            
            """
            # initialize number of successes:
            success = 0

            # for every episode:
            for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                    obs, _ = self.env.reset()      # must reset before an episode
                    done = False                   # flag is set to False initially

                    # while False:
                    while not done:
                            a = np.argmax(self.Q[obs])                     # pick best action from policy
                            obs, r, term, trunc, _ = self.env.step(a)      # step that action
                            done = term or trunc    # set to True if term or trunc

                    # if at the goal pose
                    if r == 1.0:
                            success += 1    # increment the success counter

            return round((success / num_episodes) * 100, 3)   # return success rate
    
            # average episode length:
    
    # average episode length:
    def average_length(self, num_episodes):
        """ 
        this function computes the average episode length for a given amount of episodes

        agent:          the agent that has been trained
        num_episodes:   number of episodes to play out
        returns:        the average episode length for that stretch of episodes
        
        """
        # initialize the total number of steps over the evaluation:
        total_steps = 0
        
        # for every episode:
        for _ in tqdm(range(num_episodes), colour = "#33FF00", ncols = 100):
                obs, _ = self.env.reset()      # must reset before an episode
                done = False                   # flag is set to False initially
                episode_steps = 0              # reset steps for the episode

                # while False:
                while not done:
                        a = np.argmax(self.Q[obs])                     # pick best action from policy
                        obs, _, term, trunc, _ = self.env.step(a)      # step that action
                        episode_steps += 1                             # increment episode steps

                        done = term or trunc    # set to True if term or trunc

                total_steps += episode_steps    # increment total steps by steps taken in episode
        
        # return the average steps per episode to the user:
        return round(total_steps / num_episodes, 3)
