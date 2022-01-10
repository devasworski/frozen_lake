## Task 4 ##

import numpy as np
import random


''' LinearWrapper class
        A wrapper, that allows to treat the frozen lake environment as if it would required linear action-value function aprpoximation
'''
class LinearWrapper:
    ''' __init__ function '''
    def __init__(self, env):
        self.env = env

        self.n_actions = self.env.n_actions
        self.n_states = self.env.n_states
        self.n_features = self.n_actions * self.n_states

    ''' encode_state function
        Encodes an environments state into a feature array
        
        @param s
            environment state
        
        @return features
            feature array representing the environment state
    '''
    def encode_state(self, s):
        features = np.zeros((self.n_actions, self.n_features))
        for a in range(self.n_actions):
            i = np.ravel_multi_index((s, a), (self.n_states, self.n_actions))
            features[a, i] = 1.0

        return features

    ''' decode_policy function
        Decodes a feature array into policy and value
        
        @param theta
            tolerance parameter
        
        @return policy
            the game policy
        @return value
            expected total reward starting from each game state
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
            resets the environment
        
        @return self.encode_state()
            the features of the reset environment
    '''
    def reset(self):
        return self.encode_state(self.env.reset())

    ''' step function
        takes the action
        
        @param action
            the action that will be taken

        @return self.encode_state()
            the features of the reset environment
    '''
    def step(self, action):
        state, reward, done = self.env.step(action)

        return self.encode_state(state), reward, done

    ''' render function
        calls the environment render function
        
        @param policy
            @default = None
            the policy to be used
        @param value
            @default = None
            expected total reward starting from each game state
    '''
    def render(self, policy=None, value=None):
        self.env.render(policy, value)


''' linear_sarsa function
    using sarsa algorithm combined with a linear function of features to approximate Q-function (on-policy)
    
    @param env
        current environment
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
        parameter vector with shape (action space) x (feature vector)
'''
def linear_sarsa(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)  
    eta = np.linspace(eta, 0, max_episodes)  
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)  

    for i in range(max_episodes):
        features = env.reset()  
        q = features.dot(theta) 
        if(i < env.n_actions):  a = i
        else:   a = get_action (random_state,epsilon,i,env,q)
        done = False
        while(not done):
            state_s, reward_pre, done = env.step(a) 
            state_a = get_action(random_state,epsilon,i,env,q)

            delta = reward_pre - q[a]
            q = state_s.dot(theta)
            delta += (gamma * q[state_a]) 
            theta += eta[i] * delta * features[a]
            features = state_s

            a = state_a

    return theta

''' linear_q_learning Function
    using q learning algorithm combined with a linear function of features to approximate Q-function (off-policy)
    
    @param env
        current environment
    @param max_episodes
        maximum number of episodes
    @param eta
        initial learning rate
    @param gamma
        discount factor
    @param seed
        @default = None
        seed for random numbers

    @return theta
        parameter vector with shape (action space) x (feature vector)
'''
def linear_q_learning(env, max_episodes, eta, gamma, epsilon, seed=None):
    random_state = np.random.RandomState(seed)
    eta = np.linspace(eta, 0, max_episodes)
    epsilon = np.linspace(epsilon, 0, max_episodes)

    theta = np.zeros(env.n_features)

    j = 0
    for i in range(max_episodes):
        features = env.reset()
        q = features.dot(theta)
        done = False
        while not done:

            if(j < env.n_actions):
                a = j
            else:
                a = get_action(random_state,epsilon,i,env,q,)
            j += 1

            state_s, reward_pre, done = env.step(a)

            delta = reward_pre - q[a]
            q = state_s.dot(theta)
            delta += (gamma * max(q))
            theta += eta[i] * delta * features[a] 
            features = state_s

    return theta


''' get_actions function
    get the next action for the sarsa and q-learning function

    @param env
        current environment
    @param q
        q-values
    @param epsilon
        linespace
    @param random_state
        randomState Object

    @return state_a
        the next action
'''
def get_action(random_state,epsilon,i,env,q):
    if(random_state.random(1) < epsilon[i]):
        state_a = random_state.choice(range(env.n_actions))
    else:
        state_a = random_state.choice(np.array(np.argwhere(q == np.amax(q))).flatten(), 1)[0] 
    return state_a