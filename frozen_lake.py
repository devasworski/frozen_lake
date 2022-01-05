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
        with a probability of 0.1, the action will be changed to a random action
        
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
            the averaged future reward which can be accumulated 
            by selecting actions from each game state
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
        #self.actions = [(0, 1), (-1, 0), (0, -1), (1, 0)]


        self.dict_states = {p:i for (i,p) in enumerate(product(range(self.lake.shape[0]), range(self.lake.shape[1])))}

        self.transition_popability_file = np.load('p.npy') #small lake only
                
        n_s,s,_ = self.transition_popability_file.shape

        for i in range(n_s):
            for j in range(s):
                temp = self.transition_popability_file[i,j,2]
                self.transition_popability_file[i,j,2] = self.transition_popability_file[i,j,1]
                self.transition_popability_file[i,j,1] = temp       
                
        self.transition_popability = np.zeros((self.n_states, self.n_states, self.n_actions)) #applicable to all lakes

        for current_state,current_state_index in self.dict_states.items():
                    for n, action in enumerate(self.actions):
                        
                        next_state = (current_state[0] + action[0], current_state[1] + action[1])
                        index_next_state = self.dict_states.get(next_state,'NaN')

                        if index_next_state == 'NaN': #this action would lead to a illegal state and therefore transition propability remains 0 but for for staying at the current state
                            index_next_state = current_state_index

                        current_tile = self.lake_flat[current_state_index]
                        if current_tile == '$' or current_tile == '#': #hole or goal
                            self.transition_popability[current_state_index,self.absorbing_state,n] = 1.0
                            continue

                        self.transition_popability[current_state_index,index_next_state,n] = 1.0 - self.slip + (self.slip/self.n_actions)

                        # add slip with probability of 0.1
                        for slip_n, slip_action in enumerate(self.actions):
                            if  slip_n == n: continue
                            slip_state = (current_state[0] + slip_action[0], current_state[1] + slip_action[1])
                            index_slip_state = self.dict_states.get(slip_state,'NaN')
                            if index_slip_state == 'NaN':
                                self.transition_popability[current_state_index,current_state_index,n] = self.slip/self.n_actions
                            if not index_slip_state == 'NaN':
                                self.transition_popability[current_state_index,index_slip_state,n] = self.slip/self.n_actions



        self.indices_to_states = list(product(range(self.lake.shape[0]), range(self.lake.shape[1])))
        self.states_to_indices = {s: i for (i, s) in enumerate(self.indices_to_states)}

        # A 3D cube storing the transition probabilities for each state s to each new state s' through each action a
        self.tp = np.zeros((self.n_states, self.n_states, self.n_actions))

        # Models environment deterministically
        # Modifies p values from 0 to 1 where appropriate
        for state_index, state in enumerate(self.indices_to_states):
            for state_possible_index, state_possible in enumerate(self.indices_to_states):
                for action_index, action in enumerate(self.actions):

                    # Checks if hole or goal, to only enable absorption state transitions
                    state_char = self.lake_flat[state_index]
                    if state_char == '$' or state_char == '#':
                        self.tp[state_index, n_states-1, action_index] = 1.0
                        continue

                    # Proceeds normally

                    next_state = (state[0] + action[0], state[1] + action[1])  # simulates action and gets next state
                    next_state_index = self.states_to_indices.get(next_state)  # gets index of next state coordinates

                    # If the next state is a possible state then the transition is probable
                    if next_state_index is not None and next_state_index == state_possible_index:
                        self.tp[state_index, next_state_index, action_index] = 1.0

                    # If next_state is out of bounds, default next state to current state index
                    if next_state_index is None:
                        next_state_index = self.states_to_indices.get(next_state, state_index)
                        self.tp[state_index, next_state_index, action_index] = 1.0

            # Remodels each state-state-action array to cater for slipping
            valid_states, valid_actions = np.where(self.tp[state_index] == 1)
            valid_states = np.unique(valid_states)  # At borders can have actions that map to the same state

            for state_possible_index, state_possible in enumerate(self.indices_to_states):
                for action_index, action in enumerate(self.actions):

                    # Readjust the p=1 value so that it distributes along side the slipping probabilities
                    if self.tp[state_index, state_possible_index, action_index] == 1:
                        self.tp[state_index, state_possible_index, action_index] -= self.slip

                    # if the state is reachable with other actions (hence 0), and if the action exists
                    if self.tp[state_index, state_possible_index, action_index] == 0 and \
                            state_possible_index in valid_states and action_index in valid_actions:
                        # Change p from 0 to a probability determined by slip and valid states (excluding the p=1 one)
                        self.tp[state_index, state_possible_index, action_index] = self.slip / (len(valid_states)-1)


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
        # return self.tp[state, next_state, action]
        return self.transition_popability_file[next_state, state, action]
        # nextstate state action
        
    ''' r function TODO (Definetly wrong)
        expected reward in having transitioned from state to next state given action
        
        @param next_state
            the next game state
        @param state
            current game state
        @param action
            action taken
        
        @return r
            expected reward based on parameters
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
            @default = None
            the averaged future reward which can be accumulated 
            by selecting actions from each game state 
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

