####################### IMPORTING #######################
import gymnasium as gym
import numpy as np
import random
from collections import deque

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

####################### CLASSES & FUNCTIONS #######################
# function for making a keras model:
def make_model(layers, neurons, rate, norm, drop, input_shape, output_shape, loss_function):
    # instantiate model:
    model = keras.Sequential()

    # add hidden layers:
    for i in range(layers):
        if i == 0:
            model.add(Input(shape = (input_shape, )))
            model.add(Dense(neurons, activation = 'relu', name = f'hidden_layer_{i+1}'))
        else:
            model.add(Dense(neurons, activation = 'relu', name = f'hidden_layer_{i+1}'))

        if norm == True:
            model.add(BatchNormalization(name = f'batch_norm_layer_{i+1}'))

        if drop == True:
            model.add(Dropout(0.2, name = f'dropout_layer_{i+1}'))
    
    # add output layer:
    model.add(Dense(output_shape, activation = 'linear', name = 'output_layer'))

    # compile the model:
    model.compile(optimizer = Adam(learning_rate = rate),
                  loss = loss_function)
    
    return model

# DQN agent class:
class DQN_Agent:
    ####################### INITIALIZATION #######################
    # constructor:
    def __init__(self, 
        env: gym.Env, 
        gamma: float, 
        lr: float,
        epsilon: float,
        epsilon_min: float,
        epsilon_decay: float,
        batch_size: int,
        buffer_size: int,
        target_update_freq: int, 
        layers = int,
        neurons = int):
        """ 
        this is the constructor for the agent. this agent uses a DQN to learn an optimal policy, through the use of approximator neural network 
        to approximate action-value Q, and a target network to generate a Q-target used in the updating of Q(s,a). this is done to prevent updates
        to the network weights from changing the target, meaning that we aren't bootstrapping towards a changing target. this helps to stabilize the learning.

        env:                    a gymnasium environment
        gamma:                  a float value indicating the discount factor γ
        lr:                  a float value indicating the learning rate α
        epsilon:                a float value indicating the action-selection probability ε
        epsilon_min:            a float value indicating the minimum ε value
        epsilon_decay:          a float value indicating the decay rate of ε
        batch_size:             an int representing the batch size sampled from the experience
        buffer_size:            an int representing the size of the memory buffer
        target_update_freq:     an int representing how frequently the target network weights should be updated
        layers:                 an int representing the number of layers in each network
        neurons:                an int representing the number of neurons in each network

        nS:         an int representing the number of states observed, each of which is continuous
        nA:         an int representing the number of discrete actions that can be taken

        q_network:                  a Keras sequential neural network representing the actual function approximator
        target_network:             a Keras sequential neural network representing responsible for generating Q-targets
        experience:                 an empty deque used to hold the experience history of the agent, limited to buffer_size


        """
        # object parameters:
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

        # get the environment dimensions:
        self.nS = self.env.observation_space.shape[0]
        self.nA = self.env.action_space.n

        # experience history and mini-batch size:
        self.replay_buffer = deque(maxlen = buffer_size)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.step_counter = 0

        # initialize networks:
        self.q_network = make_model(layers = layers, neurons = neurons, rate = lr,
                                                norm = True, drop = True,
                                                input_shape = self.nS, output_shape = self.nA,
                                                loss_function = 'mse')
        self.target_network = keras.models.clone_model(self.q_network)
        self.target_network.set_weights(self.q_network.get_weights())

        # set target network update frequency:
        self.target_update_freq = target_update_freq

    ####################### TRAINING #######################
    # define a decorated function to infer Q's from batched states (this is the implicit policy):
    @tf.function
    def get_qs(self, obs_batch):
        return self.q_network(obs_batch)
    
    # define a decorated function to perform the DQN training step for updating Q network weights:
    @tf.function
    def training_step(self, states, actions, rewards, next_states, dones):
        # track auto differentiation:
        with tf.GradientTape() as tape:
            # 1) do a forward pass to get Q values:
            # this is all the Q values from every state:
            q_all = self.q_network(states)

            # find relevant index of actions that will be selected:
            index = tf.stack([tf.range(tf.shape(actions)[0]), actions], axis = 1)

            # gather up the Q values that correspond to actions actually taken:
            q_selected = tf.gather_nd(q_all, index)

            # 2) compute TD-targets:
            # TD-target is computed with S', A', w-:
            q_next = self.target_network(next_states)

            # get the Q value corresponding to the max over the actions:
            max_q_next = tf.reduce_max(q_next, axis = 1)

            # compute actual TD-targets:
            targets = tf.stop_gradient(rewards + (1 - dones) * self.gamma * max_q_next)

            # 3) MSE loss between the Qs that correspond to taken actions and the TD-target:
            loss = tf.reduce_mean(tf.square(q_selected - targets))
        
        # 4) backpropagate and update the weights:
        grads = tape.gradient(loss, self.q_network.trainable_variables)
        self.q_network.optimizer.apply_gradients(zip(grads, self.q_network.trainable_variables))

    # training function:
    def training(self, training_length):

        reward_history = np.zeros(training_length)

        # for every episode:
        for episode in range(training_length):
            # reset environment:
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False

            # while false:
            while not done:
                # ε-greedy policy:
                if np.random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    obs_batch = tf.expand_dims(tf.convert_to_tensor(obs, dtype=tf.float32), 0)
                    qs = self.get_qs(obs_batch)
                    action = tf.argmax(qs[0]).numpy()

                # interact with the environment:
                next_obs, reward, term, trunc, _ = self.env.step(action)
                done = term or trunc
                self.replay_buffer.append((obs, action, reward, next_obs, done))
                obs = next_obs
                episode_reward += reward
                self.step_counter += 1

                # sample a batch of experience:
                if len(self.replay_buffer) >= self.batch_size:
                    # get a batch:
                    batch = random.sample(self.replay_buffer, self.batch_size)

                    # unpack the batch:
                    states, actions, rewards, next_states, dones = zip(*batch)

                    # convert to tensors:
                    states = tf.convert_to_tensor(states, dtype = tf.float32)
                    actions = tf.convert_to_tensor(actions, dtype = tf.int32)
                    rewards = tf.convert_to_tensor(rewards, dtype = tf.float32)
                    next_states = tf.convert_to_tensor(next_states, dtype = tf.float32)
                    dones = tf.convert_to_tensor(dones, dtype = tf.float32)

                    # single graph call:
                    self.training_step(states, actions, rewards, next_states, dones)

                    # update target network periodically:
                    if self.step_counter % self.target_update_freq == 0:
                        self.target_network.set_weights(self.q_network.get_weights())
                
            # decay epsilon:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            # advance reward history:
            reward_history[episode] = episode_reward
        
        return reward_history

