# Collaboration_Competition_Udacity
This repository contains all the necessary files for the third project of Deep Reinforcement Learning Nano-Degree by Udacity. The third project is about Deep Reinforcement Learning's part of Multi-Agent Reinforcement Learning which allows
multiple agents to collaborate and compete within an environment. In this project, a pair of agents are trained and then paired with to further collaborate and compete within an environment to play a tennis game. 

![42135619-d90f2f28-7d12-11e8-8823-82b970a54d7e](https://github.com/SAMNaqvi1212/Collaboration_Competition_Udacity/blob/main/01.gif)     

# Introduction
In this environment, two agents are tasked with controlling rackets to keep a ball in play over a net. The rules are as follows:

- If an agent successfully hits the ball over the net, it is rewarded with a score of +0.1.
- Conversely, if an agent allows the ball to hit the ground or hits the ball out of bounds, it incurs a penalty with a reward of -0.01.
- The primary objective for each agent is to maintain the ball in play and maximize their rewards.

The observation space is composed of 8 variables, representing both the position and velocity of the ball and racket. Each agent receives its own set of local observations. Additionally, there are two continuous actions available to the agents: movement toward or away from the net, and the ability to jump.

This task is episodic in nature, and to successfully solve the environment, the agents must achieve an average score of +0.5 over a span of 100 consecutive episodes, taking into account the maximum score obtained by either of the two agents in each episode. Here's how this scoring process works:

- After completing each episode, we sum up the rewards received by each agent without applying any discounting. This yields two separate scores, one for each agent.
- Subsequently, we identify the higher of these two scores to obtain a single score for that particular episode.

The ultimate goal for the agents is to attain an average score of +0.5 or higher over the course of 100 consecutive episodes, with the highest score from either agent determining their progress toward solving the environment.


# Files in this repository
Tennis.ipynb - notebook to run the project
agent.py - Multi Agent DDPG implmentation that includes code for Actor and Critic agents, functions for traing and testing the agents and utility classes such as ReplayBuffer and OUNoise
model.py - the neural network which serves as the function approximator to the DDPG Agent
checkpoint.pt - saved agent model (actor and critic)
A Report.md - document describing the solution, the learning algorithm, and ideas for future work
This README.md file

# Download the Environment

1) [Linux](https://learn.udacity.com/nanodegrees/nd893/parts/cd1764/lessons/f3f81a69-a3b4-4607-bf18-24b10e0d136a/concepts/89f15922-056f-4aed-bb8d-438503b48731)
2) [MACOSX)[https://learn.udacity.com/nanodegrees/nd893/parts/cd1764/lessons/f3f81a69-a3b4-4607-bf18-24b10e0d136a/concepts/89f15922-056f-4aed-bb8d-438503b48731]
3) [Windows32](https://learn.udacity.com/nanodegrees/nd893/parts/cd1764/lessons/f3f81a69-a3b4-4607-bf18-24b10e0d136a/concepts/89f15922-056f-4aed-bb8d-438503b48731)
4) [Windows64](https://learn.udacity.com/nanodegrees/nd893/parts/cd1764/lessons/f3f81a69-a3b4-4607-bf18-24b10e0d136a/concepts/89f15922-056f-4aed-bb8d-438503b48731)

# Optional Build your own environment
For this project, Udacity has also build an environment in Unity. One can check this on the Deep Reinforcement Learning section on Multi Agent Reinforcement Learning. 

