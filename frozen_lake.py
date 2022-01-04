## Task 1 ##

import numpy as np
from itertools import product
import contextlib

''' _printoptions function
    Configuration of the numpy print function
    
    @param *args
        <param description>
    @param *kwargs
        <param description>
'''
@contextlib.contextmanager
def _printoptions(*args, **kwargs):
    original = np.get_printoptions()
    np.set_printoptions(*args, **kwargs)
    try:
        yield
    finally:
        np.set_printoptions(**original)

''' EnvironmentModel class
        Represents a model of an environment    
'''
class EnvironmentModel:
    ''' __init__ function '''
    def __init__(self, n_states, n_actions, seed=None):
        self.n_states = n_states
        self.n_actions = n_actions

        self.random_state = np.random.RandomState(seed)

    ''' p function
        will be implemented in the child class
        
        @param next_state
            the next game state
        @param state
            current game state
        @param action
            action taken
    '''
    def p(self, next_state, state, action):
        raise NotImplementedError()

    ''' r function
        will be implemented in the child class
        
        @param next_state
            the next game state
        @param state
            current game state
        @param action
            action taken
    '''
    def r(self, next_state, state, action):
        raise NotImplementedError()

    ''' draw function
            draw the next state
        
        @param state
            current game state
        @param action
            action taken
    '''
    def draw(self, state, action):
        p = [self.p(ns, state, action) for ns in range(self.n_states)]
        next_state = self.random_state.choice(self.n_states, p=p)
        reward = self.r(next_state, state, action)

        return next_state, reward

''' Environment class
        Represents an interactive environment and inherits from EnvironmentModel     
'''
class Environment(EnvironmentModel):
    ''' __init__ function '''
    def __init__(self, n_states, n_actions, max_steps, pi, seed=None):
        EnvironmentModel.__init__(self, n_states, n_actions, seed)

        self.max_steps = max_steps

        self.pi = pi
        if self.pi is None:
            self.pi = np.full(n_states, 1. / n_states)

    ''' reset function
        reset the enviroment
        
        @return self.state
            return the reseted state
    '''
    def reset(self):
        self.n_steps = 0
        self.state = self.random_state.choice(self.n_states, p=self.pi)

        return self.state

    ''' step function
        checks the validity of the next action and makes it
        with a propability of 0.1, the action will be changed to a random action
        
        @param action
            the next action
        
        @return self.state
            return the state
        @return reward
            return the current reward
        @return done
            bool if the game max step limit has been reached 
    '''
    def step(self, action):
        #if np.random.randint(0,10,dtype=int)==9:
        #    action = self.n_actions[np.random.randint(0,4,dtype=int)]
        if action < 0 or action >= self.n_actions:
            raise Exception('Invalid action.')

        self.n_steps += 1
        done = (self.n_steps >= self.max_steps)

        self.state, reward = self.draw(self.state, action)

        return self.state, reward, done

    ''' render function TODO
        will be implemented in the child class
        
        @param policy
            @default = None 
            the current policy used
        @param value
            @default = None
            TODO <param description>
    '''
    def render(self, policy=None, value=None):
        raise NotImplementedError()

''' FrozenLake class
        Represents the frozen lake environment     
'''
class FrozenLake(Environment):
    ''' __init__ function TODO'''
    def __init__(self, lake, slip, max_steps, seed=None):
        """
        lake: A matrix that represents the lake. For example:
         lake =  [['&', '.', '.', '.'],
                  ['.', '#', '.', '#'],
                  ['.', '.', '.', '#'],
                  ['#', '.', '.', '$']]
        slip: The probability that the agent will slip
        max_steps: The maximum number of time steps in an episode
        seed: A seed to control the random number generator (optional)
        """

        # start (&), frozen (.), hole (#), goal ($)
        self.lake = np.array(lake)
        self.lake_flat = self.lake.reshape(-1)

        self.slip = slip 

        n_states = self.lake.size + 1 
        n_actions = 4

        pi = np.zeros(n_states, dtype=float) 
        pi[np.where(self.lake_flat == '&')[0]] = 1.0

        self.absorbing_state = n_states - 1

        Environment.__init__(self, n_states, n_actions, max_steps, pi, seed)

        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        self.dict_states = {p:i for (i,p) in enumerate(product(range(self.lake.shape[0]), range(self.lake.shape[1])))}

        #self.transition_popability = np.load('p.npy') #small lake only
        self.transition_popability = np.zeros((self.n_states, self.n_states, self.n_actions)) #applicable to all lakes

        for current_state,current_state_index in self.dict_states.items():
                    for n, action in enumerate(self.actions):
                        
                        next_state = (current_state[0] + action[0], current_state[1] + action[1])
                        index_next_state = self.dict_states.get(next_state,'NaN')

                        if index_next_state == 'NaN': #this action would lead to a illegal state and therefore transition propability remains 0
                            continue

                        current_tile = self.lake_flat[current_state_index]
                        if current_tile == '$' or current_tile == '#': #hole or goal
                            self.transition_popability[current_state_index,self.absorbing_state,n] = 1
                            continue

                        #non illega moves and non absorbing state:
                        self.transition_popability[current_state_index,index_next_state,n] = 1 - slip

                        # add slip with propoability of 0.1
                        for slip_n, slip_action in enumerate(self.actions):
                            if slip_n == n : continue
                            slip_state = (current_state[0] + slip_action[0], current_state[1] + slip_action[1])
                            index_slip_state = self.dict_states.get(slip_state,'NaN')
                            if index_slip_state == 'NaN': #this action would lead to a illegal state and therefore transition propability remains 0
                                self.transition_popability[current_state_index,current_state_index,slip_n] += slip/self.n_actions
                                continue
                            self.transition_popability[current_state_index,index_slip_state,slip_n] += slip/self.n_actions

    ''' step function
        calls the step function of the parent
        
        @param action
            the next action
        
        @return self.state
            return the current state
        @return reward
            return the current reward
        @return done
            bool if the game max step limit has been reached 
    '''
    def step(self, action):
        state, reward, done = Environment.step(self, action)
        done = (state == self.absorbing_state) or done
        return state, reward, done

    ''' p function
        probability to be returned for combination of next state, state, and action
        
        @param next_state
            the next game state
        @param state
            current game state
        @param action
            action taken
        
        @return p
            probability
    '''
    def p(self, next_state, state, action):
        return self.transition_popability[state, next_state, action]

    ''' r function TODO (Definetly wrong)
        <explenation>
        
        @param next_state
            the next game state
        @param state
            current game state
        @param action
            action taken
        
        @return r
            TODO <return param description>
    '''
    def r(self, next_state, state, action):
        char = 'o'
        if(state < self.n_states-1):
            char = self.lake_flat[state]
        if(char == '$'):
            return 1
        return 0

    ''' render function TODO
        renders and prints the state
        
        @param policy
            @default = None 
            the current policy used
        @param value
            TODO <explain param>
    '''
    def render(self, policy=None, value=None):
        if policy is None:
            lake = np.array(self.lake_flat)

            if self.state < self.absorbing_state:
                lake[self.state] = '@'

            print(lake.reshape(self.lake.shape))
        else:
            # UTF-8 arrows look nicer, but cannot be used in LaTeX
            # https://www.w3schools.com/charsets/ref_utf_arrows.asp
            actions = ['↑', '↓', '←', '→']

            print('Lake:')
            print(self.lake)

            print('Policy:')
            policy = np.array([actions[a] for a in policy[:-1]])
            print(policy.reshape(self.lake.shape))

            print('Value:')
            with _printoptions(precision=3, suppress=True):
                print(value[:-1].reshape(self.lake.shape))

    ''' play function
        lets the user play the game
    '''
    def play(self):
        actions = ['w', 's', 'a', 'd']

        state = self.reset()
        self.render()

        done = False
        while not done:
            c = input('\nMove: ')
            if c not in actions:
                raise Exception('Invalid action')

            state, r, done = self.step(actions.index(c))

            self.render()
            print('Reward: {0}.'.format(r))

