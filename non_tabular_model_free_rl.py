## Task 4 ##

import numpy as np
import random


''' LinearWrapper class
        A wrapper, that allows to treat the frozen lake environment as if it would required linear action-value function approximation 
'''
class LinearWrapper:
    ''' __init__ function '''
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    ''' encode_state function TODO
        TODO <explenation>
        
        @param s
        TODO <param description>
        @return features
        TODO <return param description>
    '''
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    ''' decode_policy function TODO
        TODO <explenation>
        
        @param theta
        TODO <param description>
        @return policy
        TODO <return param description>
        @return value
        TODO <return param description>
    '''
    def decode_policy(self, theta):
        policy = np.zeros(self.env.n_states, dtype=int)
        value = np.zeros(self.env.n_states)

        for s in range(self.n_states):
            features = self.encode_state(s)
            q = features.dot(theta)

            policy[s] = np.argmax(q)
            value[s] = np.max(q)

        return policy, value

    ''' reset function
        resets the enviroment
        
        @return self.encode_state()
        the features of the reseted enviroment
    '''
    def reset(self):
        return self.encode_state(self.env.reset())

    ''' step function
        takes the action
        
        @param action
        the action that will be taken

        @return self.encode_state()
        the features of the reseted enviroment
    '''
    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    ''' render function TODO
        calls the enviroment render function
        
        @param policy
        @default = None
        the policy to be used
        @param value
        @default = None
        TODO <param description>
    '''
    def render(self, policy=None, value=None):
        self.env.render(policy, value)


''' linear_sarsa function TODO
    TODO <explenation>
    
    @param env
    current enviroment
    @param max_episodes
    maximum number if episodes
    @param eta
    initial learning rate
    @param gamma
    discount factor
    @param seed
    @default = None
    seed for random numbers

    @return theta
    TODO <return param description>
'''
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)  
    eta = np.linspace(eta, 0, max_episodes)  
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)  
        
    #TODO:
    timestep = 0
    #END TODO

    for i in range(max_episodes):
        features = env.reset()  
        q = features.dot(theta) 

        #TODO:
        
        # Select action a for state s according to an e-greedy policy based on Q
        if timestep < env.n_actions:  # for the first 4 timesteps, choose each action once
            a = timestep  # select each action 0, 1, 2, 3 once
        else:
            # after having our first estimations, find the best action and break ties randomly
            best_action = randomBestAction(random_state, q)

            # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
            # (exploitation) or a random action (exploration)
            if random_state.random(1) < epsilon[i]:
                a = random_state.choice(range(env.n_actions))  # use random action
            else:
                a = best_action  # use best action
        timestep += 1

        done = False
        while not done:  # while not in absorbing state
            features_prime, r, done = env.step(a)  # Get next state and reward for the chosen action
            delta = r - q[a]  # compute the the difference between the observed reward and the estimated reward

            q = features_prime.dot(theta)  # get new estimated rewards

            # Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if timestep < env.n_actions:  # for the first 4 timesteps, choose each action once
                a_prime = timestep  # select each action 0, 1, 2, 3 once
            else:
                # after having our first estimations, find the best action and break ties randomly
                best_action = randomBestAction(random_state, q)

                # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
                # (exploitation) or a random action (exploration)
                if random_state.random(1) < epsilon[i]:
                    a_prime = random_state.choice(range(env.n_actions))  # use random action
                else:
                    a_prime = best_action  # use best action
            timestep += 1

            # Temporal difference
            delta += (gamma * q[a_prime])  # apply discount factor using the estimated value for the e-greedy policy
            theta += eta[i] * delta * features[a]  # update the weights based on gradient descent
            features = features_prime
            a = a_prime

    return theta

''' linear_q_learning Function TODO
    TODO <explenation>
    
    @param env
    current enviroment
    @param max_episodes
    maximum number if episodes
    @param eta
    initial learning rate
    @param gamma
    discount factor
    @param seed
    @default = None
    seed for random numbers

    @return theta
    TODO <return param description>
'''
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    #TODO:
    timestep = 0
    #END TODO
    
    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)

        #TODO:

        done = False
        while not done:  # while not in absorbing state

            # Select action a_prime for state s_prime according to an e-greedy policy based on Q
            if timestep < env.n_actions:  # for the first 4 timesteps, choose each action once
                a = timestep  # select each action 0, 1, 2, 3 once
            else:
                # after having our first estimations, find the best action and break ties randomly
                best_action = randomBestAction(random_state, q)

                # roll a random number from 0-1 and compare to epsilon[i] to decide whether we take best action
                # (exploitation) or a random action (exploration)
                if random_state.random(1) < epsilon[i]:
                    a = random_state.choice(range(env.n_actions))  # use random action
                else:
                    a = best_action  # use best action
            timestep += 1

            features_prime, r, done = env.step(a)  # Get next state and reward for the chosen action
            delta = r - q[a]  # compute the the difference between the observed reward and the estimated reward

            q = features_prime.dot(theta)  # get new estimated rewards
            # Temporal difference
            delta += (gamma * max(q))  # apply discount factor
            theta += eta[i] * delta * features[a]  # update the weights based on gradient descent
            features = features_prime

    return theta

''' <name> Function
    TODO: This was not part of the task description interface. Therefore we should remove it
    <explenation>
    
    @param <param name>
    <param description>
    @return <return param>
    <return param description>
'''
def randomBestAction(random_state, mean_rewards):
    # get the best actions from mean_rewards
    best_actions = np.array(np.argwhere(mean_rewards == np.amax(mean_rewards))).flatten()
    return random_state.choice(best_actions, 1)[0]  # break ties randomly and return one of the best actions
