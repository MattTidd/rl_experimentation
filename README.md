# **Reinforcement Learning Experimentation**

This repository serves to host various implementations of Reinforcement Learning (RL), as I learn and experiment with RL. 
The resources that I am using to learn RL are [David Silver's Course](https://www.youtube.com/watch?v=2pWv7GOvuf0&list=PLqYmG7hTraZDM-OYHWgPebj2MfCFzFObQ&ab_channel=GoogleDeepMind), 
as well as the 2<sup>nd</sup> edition of Sutton and Barto's [*Reinforcement Learning: an Introduction*](https://www.andrew.cmu.edu/course/10-703/textbook/BartoSutton.pdf).

# **Structure**

This implementation is built around the use of ``Gymnasium``, an open-source Python library for developing and comparing RL algorithms through the use of a standardized API.
The documentation for ``Gymnasium`` can be [found here](https://gymnasium.farama.org/introduction/basic_usage/), which highlights its introductory usage and core features. 
``Gymnasium`` is primarily built around the high-level Python class ``Env``, which approximately represents a Markov Decision Process (MDP) from RL theory.

There are several implementations of common RL environments within ``Gymnasium``, such as cartpole, pendulum, mountain-car, mujoco, atari, and more. 
As such, the structure of this repository is split based on these environments, and further subdivided by the algorithms that were implemented. The directory ``gymnasium_environments``
holds each different environment, and the algorithms that were utilized can be found within each environmental directory. 

# **Requirements**

To utilize this repository, create a virtual environment, or venv, and install the following dependencies once the venv has been activated using:

``pip install -r requirements.txt``
