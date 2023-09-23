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

# Installing Dependencies
Option # 1
1) Download it as a zip file
   
2) [click here](https://github.com/ekaakurniawan/DRLND/tree/master/p3_collab-compet) to download all the relevant files

   
Option #2
clone the repository using git version control system

'''
$ git clone https://github.com/afilimonov/udacity-deeprl-p3-collaboration-competition.git
'''

# Install Miniconda
Miniconda is a free minimal installer for conda. It is a small, bootstrap version of Anaconda that includes only conda, Python, the packages they depend on, and a small number of other useful packages, including pip, zlib, and a few others.

If you would like to know more about Anaconda, [visit this link](https://www.anaconda.com/).


In the following links, you find all the information to install Miniconda (recommended)

Download the installer:  https://docs.conda.io/en/latest/miniconda.html

Installation guide: https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Download the installer: https://www.anaconda.com/distribution/
Installation guide:  https://docs.anaconda.com/anaconda/install/

# Configure the Environment
1. Create the environment:
   
   '''
   $ conda create --name udacity-drlnd-p3 python=3.6
   $ conda activate udacity-drlnd-p3
   '''
   
3. Install PyTorch:
   Follow [this link](https://pytorch.org/get-started/locally/) to find the correct dependencies to install Pytorch

   For Linux:
   Run this command if you have an NVIDIA graphic card and want to use it
   
   '''  
   $ conda install pytorch cudatoolkit=10.1 -c pytorch
    '''
   
   Run this command, otherwise
   
   '''
   $ conda install pytorch cpuonly -c pytorch
    '''
   
   For Mac OS:
   
    '''
   $ conda install pytorch -c pytorch
     '''
 5. Install Unity Agents:

    '''
    $ pip install unityagents
    '''

# Download the Environment

1) [Linux](https://learn.udacity.com/nanodegrees/nd893/parts/cd1764/lessons/f3f81a69-a3b4-4607-bf18-24b10e0d136a/concepts/89f15922-056f-4aed-bb8d-438503b48731)
2) [MACOSX](https://learn.udacity.com/nanodegrees/nd893/parts/cd1764/lessons/f3f81a69-a3b4-4607-bf18-24b10e0d136a/concepts/89f15922-056f-4aed-bb8d-438503b48731)
3) [Windows32](https://learn.udacity.com/nanodegrees/nd893/parts/cd1764/lessons/f3f81a69-a3b4-4607-bf18-24b10e0d136a/concepts/89f15922-056f-4aed-bb8d-438503b48731)
4) [Windows64](https://learn.udacity.com/nanodegrees/nd893/parts/cd1764/lessons/f3f81a69-a3b4-4607-bf18-24b10e0d136a/concepts/89f15922-056f-4aed-bb8d-438503b48731)

# Optional Build your own environment
For this project, Udacity has also build an environment in Unity. One can check this on the Deep Reinforcement Learning section on Multi Agent Reinforcement Learning. 

