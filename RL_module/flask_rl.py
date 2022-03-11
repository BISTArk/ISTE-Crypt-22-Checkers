
################################################## 000
"""
This file contains various helper methods.
"""


import random
from pathlib import Path
import numpy as np

def flip_coin(prob):
	return random.random() < prob

def open_file(file_name, header=None):
    file = Path(file_name)
    if file.is_file():
        print(file_name, 'File exists: Enter (a)/w:')
        c = input()
        if c == "a" or c == '':
            f = open(file_name, "a")
            return f

    f = open(file_name, "w+")
    if header is not None:
        f.write(header)

    return f

def load_weights(weights_file):
    """
    weights_file: file that contains weights to be stored or loaded

    Returns: None if not to load weights else list of weights
    """

    file = Path(weights_file)
    if file.is_file():
        # print(weights_file, 'File exists: use weights:(y)/n:')
        c = 'y'
        if c == "y" or c == '':
           weights = np.loadtxt(weights_file, delimiter=',', ndmin=2)
           if len(weights) != 0:
               return weights[-1]

    return None




#######################################################  123
"""
This file contains implementation of all the agents.
"""

from abc import ABC, abstractmethod

import random


import numpy as np

CHECKERS_FEATURE_COUNT = 8
def checkers_reward(state, action, next_state):

    if next_state.is_game_over():
        # infer turn from current state, because at the end same state is used by both agents
        if state.is_first_agent_turn():
            return WIN_REWARD if next_state.is_first_agent_win() else LOSE_REWARD
        else:
            return WIN_REWARD if next_state.is_second_agent_win() else LOSE_REWARD

    agent_ind = 0 if state.is_first_agent_turn() else 1
    oppn_ind = 1 if state.is_first_agent_turn() else 0

    num_pieces_list = state.get_pieces_and_kings()

    agent_pawns = num_pieces_list[agent_ind]
    agent_kings = num_pieces_list[agent_ind + 2]

    oppn_pawns = num_pieces_list[oppn_ind]
    oppn_kings = num_pieces_list[oppn_ind + 2]

    num_pieces_list_n = next_state.get_pieces_and_kings()

    agent_pawns_n = num_pieces_list_n[agent_ind]
    agent_kings_n = num_pieces_list_n[agent_ind + 2]

    oppn_pawns_n = num_pieces_list_n[oppn_ind]
    oppn_kings_n = num_pieces_list_n[oppn_ind + 2]

    r_1 = agent_pawns - agent_pawns_n
    r_2 = agent_kings - agent_kings_n
    r_3 = oppn_pawns - oppn_pawns_n
    r_4 = oppn_kings - oppn_kings_n

    reward = r_3 * 0.2 + r_4 * 0.3 + r_1 * (-0.4) + r_2 * (-0.5)

    if reward == 0:
        reward = LIVING_REWARD

    return reward
def checkers_features(state, action):
    """
    state: game state of the checkers game
    action: action for which the feature is requested

    Returns: list of feature values for the agent whose turn is in the current state
    """
    next_state = state.generate_successor(action, False)

    agent_ind = 0 if state.is_first_agent_turn() else 1
    oppn_ind = 1 if state.is_first_agent_turn() else 0

    num_pieces_list = state.get_pieces_and_kings()

    agent_pawns = num_pieces_list[agent_ind]
    agent_kings = num_pieces_list[agent_ind + 2]
    agent_pieces = agent_pawns + agent_kings

    oppn_pawns = num_pieces_list[oppn_ind]
    oppn_kings = num_pieces_list[oppn_ind + 2]
    oppn_pieces = oppn_pawns + oppn_kings


    num_pieces_list_n = next_state.get_pieces_and_kings()

    agent_pawns_n = num_pieces_list_n[agent_ind]
    agent_kings_n = num_pieces_list_n[agent_ind + 2]
    agent_pieces_n = agent_pawns_n + agent_kings_n

    oppn_pawns_n = num_pieces_list_n[oppn_ind]
    oppn_kings_n = num_pieces_list_n[oppn_ind + 2]
    oppn_pieces_n = oppn_pawns_n + oppn_kings_n

    features = []

    # features.append(agent_pawns_n - agent_pawns)
    # features.append(agent_kings_n - agent_kings)
    # features.append(agent_pieces_n - agent_pieces)

    # pawns and kings of agent and opponent in current state
    features.append(agent_pawns)
    features.append(agent_kings)
    features.append(oppn_pawns)
    features.append(oppn_kings)

    features.append(oppn_pawns_n - oppn_pawns)
    features.append(oppn_kings_n - oppn_kings)
    features.append(oppn_pieces_n - oppn_pieces)

    features.append(next_state.num_attacks())

    # print(features)
    return features



class Agent(ABC):

    def __init__(self, is_learning_agent=False):
        self.is_learning_agent = is_learning_agent
        self.has_been_learning_agent = is_learning_agent

    @abstractmethod
    def get_action(self, state):
        """
        state: the state in which to take action
        Returns: the single action to take in this state
        """
        pass


class KeyBoardAgent(Agent):

    def __init__(self):
        Agent.__init__(self)


    def get_action(self, state):
        """
        state: the current state from which to take action

        Returns: list of starting position, ending position
        """

        start = [int(pos) for pos in input("Enter start position (e.g. x y): ").split(" ")]
        end = [int(pos) for pos in input("Enter end position (e.g. x y): ").split(" ")]

        ends = []
        i=1
        while i < len(end):
            ends.append([end[i-1], end[i]])
            i += 2

        action = [start] + ends
        return action


class AlphaBetaAgent(Agent):

    def __init__(self, depth):
        Agent.__init__(self, is_learning_agent=False)
        self.depth = depth

    def evaluation_function(self, state, agent=True):
        """
        state: the state to evaluate
        agent: True if the evaluation function is in favor of the first agent and false if
               evaluation function is in favor of second agent

        Returns: the value of evaluation
        """
        agent_ind = 0 if agent else 1
        other_ind = 1 - agent_ind

        if state.is_game_over():
            if agent and state.is_first_agent_win():
                return 500

            if not agent and state.is_second_agent_win():
                return 500

            return -500

        pieces_and_kings = state.get_pieces_and_kings()
        return pieces_and_kings[agent_ind] + 2 * pieces_and_kings[agent_ind + 2] - \
        (pieces_and_kings[other_ind] + 2 * pieces_and_kings[other_ind + 2])

    def get_action(self, state):

        def mini_max(state, depth, agent, A, B):
            if agent >= state.get_num_agents():
                agent = 0

            depth += 1
            if depth == self.depth or state.is_game_over():
                return [None, self.evaluation_function(state, max_agent)]
            elif agent == 0:
                return maximum(state, depth, agent, A, B)
            else:
                return minimum(state, depth, agent, A, B)

        def maximum(state, depth, agent, A, B):
            output = [None, -float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent + 1, A, B)

                check = val[1]

                if check > output[1]:
                    output = [action, check]

                if check > B:
                    return [action, check]

                A = max(A, check)

            return output

        def minimum(state, depth, agent, A, B):
            output = [None, float("inf")]
            actions_list = state.get_legal_actions()

            if not actions_list:
                return [None, self.evaluation_function(state, max_agent)]

            for action in actions_list:
                current = state.generate_successor(action)
                val = mini_max(current, depth, agent+1, A, B)

                check = val[1]

                if check < output[1]:
                    output = [action, check]

                if check < A:
                    return [action, check]

                B = min(B, check)

            return output

        # max_agent is true meaning it is the turn of first player at the state in 
        # which to choose the action
        max_agent = state.is_first_agent_turn()
        output = mini_max(state, -1, 0, -float("inf"), float("inf"))
        return output[0]


class ReinforcementLearningAgent(Agent):

    def __init__(self, is_learning_agent=True):
        Agent.__init__(self, is_learning_agent)

        self.episodes_so_far = 0


    @abstractmethod
    def get_action(self, state):
        """
        state: the current state from which to take action

        Returns: the action to perform
        """
        # TODO call do_action from this method
        pass


    @abstractmethod
    def update(self, state, action, next_state, reward):
        """
        performs update for the learning agent

        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        pass

    def start_episode(self):
        # Accumulate rewards while training for each episode and show total rewards 
        # at the end of each episode i.e. when stop episode
        self.prev_state = None
        self.prev_action = None

        self.episode_rewards = 0.0


    def stop_episode(self):
        # print('reward this episode', self.episode_rewards)
        pass

    @abstractmethod
    def start_learning(self):
        pass


    @abstractmethod
    def stop_learning(self):
        pass


    @abstractmethod
    def observe_transition(self, state, action, next_state, reward, next_action=None):
        pass


    @abstractmethod
    def observation_function(self, state):
        pass


    # TODO
    def reward_function(self, state, action, next_state):
        # make a reward function for the environment
        return checkers_reward(state, action, next_state)


    def do_action(self, state, action):
        """
        called by get_action to update previous state and action
        """
        self.prev_state = state
        self.prev_action = action


class QLearningAgent(ReinforcementLearningAgent):

    def __init__(self, alpha=0.01, gamma=0.1, epsilon=0.5, is_learning_agent=True, weights=None):

        """
        alpha: learning rate
        gamma: discount factor
        epsilon: exploration constant
        is_learning_agent: whether to treat this agent as learning agent or not
        weights: default weights
        """

        ReinforcementLearningAgent.__init__(self, is_learning_agent=is_learning_agent)

        self.original_alpha = alpha
        self.original_epsilon = epsilon

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        if not is_learning_agent:
            self.epsilon = 0.0
            self.alpha = 0.0


        if weights is None:
            # initialize weights for the features
            self.weights = np.zeros(CHECKERS_FEATURE_COUNT)
        else:
            if len(weights) != CHECKERS_FEATURE_COUNT:
                raise Exception("Invalid weights " + weights)

            self.weights = np.array(weights, dtype=float)


    def start_learning(self):
        """
        called by environment to notify agent of starting new episode
        """

        self.alpha = self.original_alpha
        self.epsilon = self.original_epsilon

        self.is_learning_agent = True


    def stop_learning(self):
        """
        called by environment to notify agent about end of episode
        """
        self.alpha = 0.0
        self.epsilon = 0.0

        self.is_learning_agent = False


    def get_q_value(self, state, action, features):
        """
          Returns: Q(state,action)
        """
        q_value = np.dot(self.weights, features)
        return q_value


    def compute_value_from_q_values(self, state):
        """
          Returns: max_action Q(state, action) where the max is over legal actions.
                   If there are no legal actions, which is the case at the terminal state, 
                   return a value of 0.0.
        """
        actions = state.get_legal_actions()

        if not actions:
            return 0.0

        q_values = \
        [self.get_q_value(state, action, checkers_features(state, action)) for action in actions]

        return max(q_values)


    def compute_action_from_q_values(self, state, actions):
        """
          Returns: the best action to take in a state. If there are no legal actions,
                   which is the case at the terminal state, return None.
        """
        if not actions:
            return None

        # if max_value < 0:
        #     return random.choice(actions)

        arg_max = np.argmax([self.get_q_value(state, action, checkers_features(state, action)) 
            for action in actions])

        return actions[arg_max]


    def get_action(self, state):
        """
          Returns: the action to take in the current state.  With probability self.epsilon,
                   take a random action and take the best policy action otherwise.  If there are
                   no legal actions, which is the case at the terminal state, returns None.
        """

        # Pick Action
        legal_actions = state.get_legal_actions()
        action = None

        if not legal_actions:
            return None

        if flip_coin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.compute_action_from_q_values(state, legal_actions)

        self.do_action(state, action)
        return action


    def update(self, state, action, next_state, reward):

        features = checkers_features(state, action)

        expected = reward + self.gamma * self.compute_value_from_q_values(next_state)
        current = self.get_q_value(state, action, features)

        temporal_difference = expected - current

        for i in range(CHECKERS_FEATURE_COUNT):
            self.weights[i] = self.weights[i] + self.alpha * (temporal_difference) * features[i]


    def getPolicy(self, state):
        return self.compute_action_from_q_values(state, state.get_legal_actions())


    def getValue(self, state):
        return self.compute_value_from_q_values(state)  


    def observe_transition(self, state, action, next_state, reward, next_action=None):
        """
        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        self.episode_rewards += reward
        self.update(state, action, next_state, reward)


    def observation_function(self, state):
        if self.prev_state is not None:
            reward = self.reward_function(self.prev_state, self.prev_action, state)
            # print('reward is', reward)
            self.observe_transition(self.prev_state, self.prev_action, state, reward)

    def update_parameters(self, freq, num_games):
        if num_games % freq == 0:
            self.original_alpha /= 2.0
            self.original_epsilon /= 2.0


class SarsaLearningAgent(QLearningAgent):

    def __init__(self, alpha=0.01, gamma=0.1, epsilon=0.5, is_learning_agent=True, weights=None):
        
        QLearningAgent.__init__(self, alpha, gamma, epsilon, is_learning_agent, weights)


    def update(self, state, action, next_state, next_action, reward):

        features = checkers_features(state, action)

        if next_action is None:
            next_q_value = 0.0
        else:
            next_q_value = \
            self.get_q_value(next_state, next_action, checkers_features(next_state, next_action))
    
        expected = reward + self.gamma * next_q_value

        current = self.get_q_value(state, action, features)

        temporal_difference = expected - current

        for i in range(CHECKERS_FEATURE_COUNT):
            self.weights[i] = self.weights[i] + self.alpha * (temporal_difference) * features[i]


    def observe_transition(self, state, action, next_state, next_action, reward):
        """
        state: the state (s) in which action was taken
        action: the action (a) taken in the state (s)
        next_state: the next state (s'), in which agnet will perform next action, 
                    that resulted from state (s) and action (a)
        reward: reward obtained for taking action (a) in state (s) and going to next state (s')
        """
        self.episode_rewards += reward
        self.update(state, action, next_state, next_action, reward)


    def observation_function(self, state):
        if self.prev_state is not None:
            reward = self.reward_function(self.prev_state, self.prev_action, state)
            # print('reward is', reward)
            action = self.get_action(state)
            self.observe_transition(self.prev_state, self.prev_action, state, action, reward)

            return action


class SarsaSoftmaxAgent(SarsaLearningAgent):

    def __init__(self, alpha=0.01, gamma=0.1, t=1.0, is_learning_agent=True, weights=None):
        SarsaLearningAgent.__init__(self, alpha=alpha, gamma=gamma,
            is_learning_agent=is_learning_agent, weights=weights)

        self.t = t

    def get_action(self, state):
        legal_actions = state.get_legal_actions()

        if not legal_actions:
            return None

        if self.epsilon == 0.0:
            return self.compute_action_from_q_values(state, legal_actions)

        q_values = [self.get_q_value(state, action, checkers_features(state, action))
                for action in legal_actions]

        exps = np.exp(q_values) / self.t
        probs = exps / np.sum(exps)

        action_ind = np.random.choice(len(legal_actions), p=probs)

        self.do_action(state, legal_actions[action_ind])
        return legal_actions[action_ind]

    def update_parameters(self, freq, num_games):
        if num_games % freq == 0:
            self.t /= 2.0

############################################################## 456

"""
This file contains implementation of checkers game.
This file also contains feature, reward functions and
methods to run a single game.
"""

import math
import copy
from functools import reduce


CHECKERS_FEATURE_COUNT = 8
WIN_REWARD = 500
LOSE_REWARD = -500
LIVING_REWARD = -0.1

def getmove():
    print(WIN_REWARD)

def getRLMove(board):
    board
    print('')

def func():
    print('')
class Board:

    """
    A class to represent and play an 8x8 game of checkers.
    """
    EMPTY_SPOT = 0
    P1 = 1
    P2 = 2
    P1_K = 3
    P2_K = 4
    BACKWARDS_PLAYER = P2
    HEIGHT = 8
    WIDTH = 4

    P1_SYMBOL = 'o'
    P1_K_SYMBOL = 'O'
    P2_SYMBOL = 'x'
    P2_K_SYMBOL = 'X'


    def __init__(self, old_spots=None, the_player_turn=True):
        """
        Initializes a new instance of the Board class.  Unless specified otherwise, 
        the board will be created with a start board configuration.

        the_player_turn=True indicates turn of player P1

        NOTE:
        Maybe have default parameter so board is 8x8 by default but nxn if wanted.
        """
        self.player_turn = the_player_turn
        if old_spots is None:
            self.spots = [[j, j, j, j] for j in [self.P1, self.P1, self.P1, self.EMPTY_SPOT, 
                                                self.EMPTY_SPOT, self.P2, self.P2, self.P2]]
        else:
            self.spots = old_spots


    def reset_board(self):
        """
        Resets the current configuration of the game board to the original 
        starting position.
        """
        self.spots = Board().spots


    def empty_board(self):
        """
        Removes any pieces currently on the board and leaves the board with nothing but empty spots.
        """
        # TODO Make sure [self.EMPTY_SPOT]*self.HEIGHT] has no issues
        self.spots = [[j, j, j, j] for j in [self.EMPTY_SPOT] * self.HEIGHT]   

    
    def is_game_over(self):
        """
        Finds out and returns weather the game currently being played is over or
        not.
        """
        if not self.get_possible_next_moves():
            return True

        return False


    def not_spot(self, loc):
        """
        Finds out of the spot at the given location is an actual spot on the game board.
        """
        if len(loc) == 0 or loc[0] < 0 or loc[0] > self.HEIGHT - 1 or loc[1] < 0 or \
            loc[1] > self.WIDTH - 1:
            return True
        return False


    def get_spot_info(self, loc):
        """
        Gets the information about the spot at the given location.
        
        NOTE:
        Might want to not use this for the sake of computational time.
        """
        return self.spots[loc[0]][loc[1]]


    def forward_n_locations(self, start_loc, n, backwards=False):
        """
        Gets the locations possible for moving a piece from a given location diagonally
        forward (or backwards if wanted) a given number of times(without directional change midway).
        """
        if n % 2 == 0:
            temp1 = 0
            temp2 = 0
        elif start_loc[0] % 2 == 0:
            temp1 = 0
            temp2 = 1 
        else:
            temp1 = 1
            temp2 = 0

        answer = [[start_loc[0], start_loc[1] + math.floor(n / 2) + temp1], 
                    [start_loc[0], start_loc[1] - math.floor(n / 2) - temp2]]

        if backwards: 
            answer[0][0] = answer[0][0] - n
            answer[1][0] = answer[1][0] - n
        else:
            answer[0][0] = answer[0][0] + n
            answer[1][0] = answer[1][0] + n

        if self.not_spot(answer[0]):
            answer[0] = []
        if self.not_spot(answer[1]):
            answer[1] = []

        return answer


    def get_simple_moves(self, start_loc):
        """
        Gets the possible moves a piece can make given that it does not capture any 
        opponents pieces.

        PRE-CONDITION:
        -start_loc is a location with a players piece
        """
        if self.spots[start_loc[0]][start_loc[1]] > 2:
            next_locations = self.forward_n_locations(start_loc, 1)
            next_locations.extend(self.forward_n_locations(start_loc, 1, True))
        elif self.spots[start_loc[0]][start_loc[1]] == self.BACKWARDS_PLAYER:
            next_locations = self.forward_n_locations(start_loc, 1, True)  
        else:
            next_locations = self.forward_n_locations(start_loc, 1)
        

        possible_next_locations = []

        for location in next_locations:
            if len(location) != 0:
                if self.spots[location[0]][location[1]] == self.EMPTY_SPOT:
                    possible_next_locations.append(location)
            
        return [[start_loc, end_spot] for end_spot in possible_next_locations]


    def get_capture_moves(self, start_loc, move_beginnings=None):
        """
        Recursively get all of the possible moves for a piece which involve capturing an 
        opponent's piece.
        """
        if move_beginnings is None:
            move_beginnings = [start_loc]
            
        answer = []
        if self.spots[start_loc[0]][start_loc[1]] > 2:  
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)
            next1.extend(self.forward_n_locations(start_loc, 1, True))
            next2.extend(self.forward_n_locations(start_loc, 2, True))
        elif self.spots[start_loc[0]][start_loc[1]] == self.BACKWARDS_PLAYER:
            next1 = self.forward_n_locations(start_loc, 1, True)
            next2 = self.forward_n_locations(start_loc, 2, True)
        else:
            next1 = self.forward_n_locations(start_loc, 1)
            next2 = self.forward_n_locations(start_loc, 2)
        
        
        for j in range(len(next1)):
            # if both spots exist
            if (not self.not_spot(next2[j])) and (not self.not_spot(next1[j])) : 
                # if next spot is opponent
                if self.get_spot_info(next1[j]) != self.EMPTY_SPOT and \
                    self.get_spot_info(next1[j]) % 2 != self.get_spot_info(start_loc) % 2:  
                    # if next next spot is empty
                    if self.get_spot_info(next2[j]) == self.EMPTY_SPOT:
                        temp_move1 = copy.deepcopy(move_beginnings)
                        temp_move1.append(next2[j])
                        
                        answer_length = len(answer)
                        
                        if self.get_spot_info(start_loc) != self.P1 or \
                            next2[j][0] != self.HEIGHT - 1: 
                            if self.get_spot_info(start_loc) != self.P2 or next2[j][0] != 0: 

                                temp_move2 = [start_loc, next2[j]]
                                
                                temp_board = Board(copy.deepcopy(self.spots), self.player_turn)
                                temp_board.make_move(temp_move2, False)

                                answer.extend(temp_board.get_capture_moves(temp_move2[1], temp_move1))
                                
                        if len(answer) == answer_length:
                            answer.append(temp_move1)
                            
        return answer


    def get_piece_locations(self):
        """
        Gets all the pieces of the current player
        """
        piece_locations = []
        for j in range(self.HEIGHT):
            for i in range(self.WIDTH):
                if (self.player_turn == True and 
                    (self.spots[j][i] == self.P1 or self.spots[j][i] == self.P1_K)) or \
                (self.player_turn == False and 
                    (self.spots[j][i] == self.P2 or self.spots[j][i] == self.P2_K)):
                    piece_locations.append([j, i])  

        return piece_locations        
    
    
        

    def get_possible_next_moves(self):
        """
        Gets the possible moves that can be made from the current board configuration.
        """

        # for move in MOVES:
        #     for single_move in move:
        #         single_move[1] = 3 - single_move[1]
        
        piece_locations = self.get_piece_locations()

        try:  #Should check to make sure if this try statement is still necessary 
            capture_moves = list(reduce(lambda a, b: a + b, list(map(self.get_capture_moves, piece_locations))))  # CHECK IF OUTER LIST IS NECESSARY

            if len(capture_moves) != 0:
                return capture_moves

            
            MOVES = list(reduce(lambda a, b: a + b, list(map(self.get_simple_moves, piece_locations))))  # CHECK IF OUTER LIST IS NECESSARY 
            
            return MOVES
            
        except TypeError:
            return []
    
    def get_board(self):
        board = [[0 for x in range(8)] for y in range(8)] 

        

        for i in range(8):
            for j in range(8):
                if( (i + j) % 2 == 0):
                    board[i][j] = self.spots[int(i)][int(j/2)] 
                else:
                    board[i][j] = 0

        for i in board:
            print(i)

    def get_compressed_board(self, board):
        compressed_board = [[0 for x in range(8)] for y in range(4)]

        for i in range(8):
            compressed_board_row = []
            for j in range(8):
                if( (i + j) % 2 == 0):
                    compressed_board_row[i][j]

    def show_spots(self):
        for i in self.spots:
            print(i)
                

    def make_move(self, move, switch_player_turn=True):
        """
        Makes a given move on the board, and (as long as is wanted) switches the indicator for
        which players turn it is.
        """
        print("This is move: ")
        board = self.spots
        
        print(move)
        # print(board)
        
        if abs(move[0][0] - move[1][0]) == 2:
            for j in range(len(move) - 1):
                if move[j][0] % 2 == 1:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j][1]
                    else:
                        middle_y = move[j + 1][1]
                else:
                    if move[j + 1][1] < move[j][1]:
                        middle_y = move[j + 1][1]
                    else:
                        middle_y = move[j][1]
                        
                self.spots[int((move[j][0] + move[j + 1][0]) / 2)][middle_y] = self.EMPTY_SPOT


        self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.spots[move[0][0]][move[0][1]]
        if move[len(move) - 1][0] == self.HEIGHT - 1 and self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.P1:
            self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.P1_K
        elif move[len(move) - 1][0] == 0 and self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] == self.P2:
            self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.P2_K
        else:
            self.spots[move[len(move) - 1][0]][move[len(move) - 1][1]] = self.spots[move[0][0]][move[0][1]]
        self.spots[move[0][0]][move[0][1]] = self.EMPTY_SPOT

        if switch_player_turn:
            self.player_turn = not self.player_turn

        # self.show_spots()
        # self.get_board()

    def get_potential_spots_from_moves(self, moves):
        """
        Get's the potential spots for the board if it makes any of the given moves.
        If moves is None then returns it's own current spots.
        """
        if moves is None:
            return self.spots
        answer = []
        for move in moves:
            original_spots = copy.deepcopy(self.spots)
            self.make_move(move, switch_player_turn=False)
            answer.append(self.spots) 
            self.spots = original_spots 
        
        return answer

    def insert_pieces(self, pieces_info):
        """
        Inserts a set of pieces onto a board.
        pieces_info is in the form: [[vert1, horz1, piece1], [vert2, horz2, piece2], ..., [vertn, horzn, piecen]]
        """
        for piece_info in pieces_info:
            self.spots[piece_info[0]][piece_info[1]] = piece_info[2]


    def get_symbol(self, location):
        """
        Gets the symbol for what should be at a board location.
        """
        if self.spots[location[0]][location[1]] == self.EMPTY_SPOT:
            return " "
        elif self.spots[location[0]][location[1]] == self.P1:
            return self.P1_SYMBOL
        elif self.spots[location[0]][location[1]] == self.P2:
            return self.P2_SYMBOL
        elif self.spots[location[0]][location[1]] == self.P1_K:
            return self.P1_K_SYMBOL
        else:
            return self.P2_K_SYMBOL


    def print_reflected_board(self): #todo
        a = self.spots
        for i in range(8):
            for j in range(4):
                a[i][j] = self.spots[i][3-j]

        self.spots = a

    def flip_board(self):
        
        a = self.print_board()
        print()
        for i in range(8):
            for j in range(4):
                a[i][j], a[i][7 - j] = a[i][7 - j], a[i][j]
                

        for i in range(8):
            for j in range(8):
                if(a[i][j] == ' '):
                    a[i][j] = 0
                if(a[i][j] == 'o'):
                    a[i][j] = 2
                if(a[i][j] == 'O'):
                    a[i][j] = 4
                if(a[i][j] == 'x'):
                    a[i][j] = 1
                if(a[i][j] == 'X'):
                    a[i][j] = 3
        for i in a:
            print(i)

        return a

    def print_board(self):
        """
        Prints a string representation of the current game board.
        """
        # self.print_reflected_board()
        
        # internal_board = [[0 for x in range(8)] for y in range(8)]
        internal_board = [[]]
        

        index_columns = "   "
        for j in range(self.WIDTH):
            index_columns += " " + str(j) + "   " + str(j) + "  "
        print(index_columns)

        norm_line = "  |---|---|---|---|---|---|---|---|"
        print(norm_line)
        row = []
        for j in range(self.HEIGHT):
            temp_line = str(j) + " "
            if j % 2 == 1:
                temp_line += "|///|"
                row.append(0)
            else:
                temp_line += "|"
            
            for i in range(self.WIDTH):
                temp_line = temp_line + " " + self.get_symbol([j, i]) + " |"
                row.append(self.get_symbol([j, i]))
                if i != 3 or j % 2 != 1:  # TODO should figure out if this 3 should be changed to self.WIDTH-1
                    temp_line = temp_line + "///|"
                    row.append(0)
            print(temp_line)
            print(norm_line)
            internal_board.append(row)
            
            row = []

        for i in internal_board[1:]:
            print(i)

        # self.flip_board()
        return internal_board[1:]
            



def checkers_features(state, action):
    """
    state: game state of the checkers game
    action: action for which the feature is requested

    Returns: list of feature values for the agent whose turn is in the current state
    """
    next_state = state.generate_successor(action, False)

    agent_ind = 0 if state.is_first_agent_turn() else 1
    oppn_ind = 1 if state.is_first_agent_turn() else 0

    num_pieces_list = state.get_pieces_and_kings()

    agent_pawns = num_pieces_list[agent_ind]
    agent_kings = num_pieces_list[agent_ind + 2]
    agent_pieces = agent_pawns + agent_kings

    oppn_pawns = num_pieces_list[oppn_ind]
    oppn_kings = num_pieces_list[oppn_ind + 2]
    oppn_pieces = oppn_pawns + oppn_kings


    num_pieces_list_n = next_state.get_pieces_and_kings()

    agent_pawns_n = num_pieces_list_n[agent_ind]
    agent_kings_n = num_pieces_list_n[agent_ind + 2]
    agent_pieces_n = agent_pawns_n + agent_kings_n

    oppn_pawns_n = num_pieces_list_n[oppn_ind]
    oppn_kings_n = num_pieces_list_n[oppn_ind + 2]
    oppn_pieces_n = oppn_pawns_n + oppn_kings_n

    features = []

    # features.append(agent_pawns_n - agent_pawns)
    # features.append(agent_kings_n - agent_kings)
    # features.append(agent_pieces_n - agent_pieces)

    # pawns and kings of agent and opponent in current state
    features.append(agent_pawns)
    features.append(agent_kings)
    features.append(oppn_pawns)
    features.append(oppn_kings)

    features.append(oppn_pawns_n - oppn_pawns)
    features.append(oppn_kings_n - oppn_kings)
    features.append(oppn_pieces_n - oppn_pieces)

    features.append(next_state.num_attacks())

    # print(features)
    return features


def checkers_reward(state, action, next_state):

    if next_state.is_game_over():
        # infer turn from current state, because at the end same state is used by both agents
        if state.is_first_agent_turn():
            return WIN_REWARD if next_state.is_first_agent_win() else LOSE_REWARD
        else:
            return WIN_REWARD if next_state.is_second_agent_win() else LOSE_REWARD

    agent_ind = 0 if state.is_first_agent_turn() else 1
    oppn_ind = 1 if state.is_first_agent_turn() else 0

    num_pieces_list = state.get_pieces_and_kings()

    agent_pawns = num_pieces_list[agent_ind]
    agent_kings = num_pieces_list[agent_ind + 2]

    oppn_pawns = num_pieces_list[oppn_ind]
    oppn_kings = num_pieces_list[oppn_ind + 2]

    num_pieces_list_n = next_state.get_pieces_and_kings()

    agent_pawns_n = num_pieces_list_n[agent_ind]
    agent_kings_n = num_pieces_list_n[agent_ind + 2]

    oppn_pawns_n = num_pieces_list_n[oppn_ind]
    oppn_kings_n = num_pieces_list_n[oppn_ind + 2]

    r_1 = agent_pawns - agent_pawns_n
    r_2 = agent_kings - agent_kings_n
    r_3 = oppn_pawns - oppn_pawns_n
    r_4 = oppn_kings - oppn_kings_n

    reward = r_3 * 0.2 + r_4 * 0.3 + r_1 * (-0.4) + r_2 * (-0.5)

    if reward == 0:
        reward = LIVING_REWARD

    return reward


class Game:
    """
    A class to control a game by asking for actions from agents while following game rules.
    """

    def __init__(self, first_agent, second_agent, game_state, rules):
        """
        first_agent: first agent which corresponds to board.player_turn = True
        second_agent: second agent other than first agent
        game_state: state of the game an instance of GameState
        rules: an instance of ClassicGameRules
        """

        self.first_agent = first_agent
        self.second_agent = second_agent
        self.game_state = game_state
        self.rules = rules

    def run_move(self, board=None):
        quiet = self.rules.quiet
        game_state = self.game_state


        action = None
        num_moves = 0
        if(not game_state.is_game_over() and num_moves < self.rules.max_moves):   
            # get the agent whose turn is next
            # print('number of pieces', game_state.get_pieces_and_kings(True), game_state.get_pieces_and_kings(False))
            active_agent = self.first_agent if game_state.is_first_agent_turn() else self.second_agent


            if not quiet:
                game_state.board.show_spots()
                game_state.print_board()
                # game_state.flip_board()
                print('Current turn is of agent: ' + str(game_state.player_symbol(game_state.player_info())))
                print('Available moves: ' + str(game_state.get_legal_actions()))
                # game_state.num_attacks()
                # input()

            if action is None:
                action = active_agent.get_action(game_state)
                
            print('ACTION')
            print(action)
            
        return action
        #     next_game_state = game_state.generate_successor(action)
        #     self.game_state = next_game_state

        #     game_state = self.game_state
            
        #     #after RL agent plays
        #     # game_state.board.show_spots()
            
        #     num_moves += 1
        #     # input()

        # if num_moves >= self.rules.max_moves:
        #     game_state.set_max_moves_done()

        # # after the game is over, tell learning agents to learn accordingly

        # # inform learning agents about new episode end
        # for learning_agent in learning_agents:
        #     learning_agent.observation_function(game_state)
        #     learning_agent.stop_episode()

        

        # return num_moves, game_state

    def run(self):

        quiet = self.rules.quiet
        game_state = self.game_state

        learning_agents = []

        if self.first_agent.is_learning_agent:
            learning_agents.append(self.first_agent)

        if self.second_agent.is_learning_agent:
            learning_agents.append(self.second_agent)

        # inform learning agents about new episode start
        for learning_agent in learning_agents:
            learning_agent.start_episode()


        action = None
        num_moves = 0
        while not game_state.is_game_over() and num_moves < self.rules.max_moves:
            # get the agent whose turn is next
            # print('number of pieces', game_state.get_pieces_and_kings(True), game_state.get_pieces_and_kings(False))
            active_agent = self.first_agent if game_state.is_first_agent_turn() else self.second_agent

            if active_agent.is_learning_agent:
                action = active_agent.observation_function(game_state)
            else:
                action = None

            if not quiet:
                game_state.board.show_spots()
                game_state.print_board()
                # game_state.flip_board()
                print('Current turn is of agent: ' + str(game_state.player_symbol(game_state.player_info())))
                print('Available moves: ' + str(game_state.get_legal_actions()))
                # game_state.num_attacks()
                # input()

            if action is None:
                action = active_agent.get_action(game_state)
                
            print('ACTION')
            print(action)
            
            next_game_state = game_state.generate_successor(action)
            self.game_state = next_game_state

            game_state = self.game_state
            
            #after RL agent plays
            # game_state.board.show_spots()
            
            num_moves += 1
            # input()

        if num_moves >= self.rules.max_moves:
            game_state.set_max_moves_done()

        # after the game is over, tell learning agents to learn accordingly

        # inform learning agents about new episode end
        for learning_agent in learning_agents:
            learning_agent.observation_function(game_state)
            learning_agent.stop_episode()

        

        return num_moves, game_state


##########################################################  789

"""
This file contains code to handle game play
"""

import copy
import sys
import csv
import time
import traceback
from collections import deque
from multiprocessing import Pool

# from util import open_file, load_weights

# # from game import *
# from agents import *

import numpy as np

def func():
    print('')

#number of weights to remember
NUM_WEIGHTS_REM = 5
WEIGHTS_SAVE_FREQ = 50
WRITE_FREQ = 100
TEST_FREQ = 100
TEST_GAMES = 100
NOTIFY_FREQ = 50
CHANGE_AGENT_FREQ = 10

class GameState:
    """
    A class which stores information about the state of a game.
    This class uses class Board to perform moves and to check whether game is won or lost.
    """


    def __init__(self, prev_state=None, the_player_turn=True):
        """
        prev_state: an instance of GameState or None
        """

        if prev_state is None:
            prev_spots = None
        else:
            prev_spots = copy.deepcopy(prev_state.board.spots)

        self.board = Board(prev_spots, the_player_turn)
        self.max_moves_done = False

    def get_num_agents(self):
        return 2

    def get_legal_actions(self):
        """
        Returns the legal moves as list of moves. A single move is a list of positions going from
        first position to next position
        """
        return self.board.get_possible_next_moves()


    def generate_successor(self, action, switch_player_turn=True):
        """
        action is a list of positions indicating move from position at first index to position at
        next index

        Returns: a new state without any changes to current state
        """

        successor_state = GameState(self, self.board.player_turn)
        successor_state.board.make_move(action, switch_player_turn)

        return successor_state

    def is_first_agent_turn(self):
        """
        Returns: True if it is the turn of first agent else returns False
        """
        return self.board.player_turn


    def is_game_over(self):
        """
        Returns: True if either agent has won the game
        """
        return self.board.is_game_over() or self.max_moves_done

    def is_first_agent_win(self):
        """
        Returns: False if game is still on or first agent has lost and True iff first agent has won
        """

        # If max moves has reached, none of the agents has won
        if self.max_moves_done:
            return False

        if not self.is_game_over() or self.is_first_agent_turn():
            return False

        return True

    def is_second_agent_win(self):
        """
        Returns: False if game is still on or second agent has lost and True iff second agent has won
        """

        # If max moves has reached, none of the agents has won
        if self.max_moves_done:
            return False

        if not self.is_game_over() or not self.is_first_agent_turn():
            return False

        return True

    def flip_board(self):
        
        return self.board.flip_board()

    def print_board(self):
        self.board.print_board()


    def player_info(self):
        """
        Returns: the index of player (P1 or P2) whose turn is next
        """

        # if player_turn is true, it indicates turn of player P1
        return self.board.P1 if self.board.player_turn else self.board.P2


    def player_symbol(self, index):
        """
        index: index of the player to be queried 1 or 2

        Returns: symbol corresponding to the player in the game
        """
        if index == 1:
            return self.board.P1_SYMBOL
        else:
            return self.board.P2_SYMBOL

    def show_spots(self):
        for i in self.board.spots:
            print(i)

    def get_pieces_and_kings(self, player=None):
        """
        player: True if for the first player, false for the second player, None for both players

        Returns: the number of pieces and kings for every player in the current state
        """
        spots = self.board.spots

        # first agent pawns, second agent pawns, first agent kings, second agent kings
        count = [0,0,0,0]   
        for x in spots:
            for y in x:
                if y != 0:
                    count[y-1] = count[y-1] + 1

        if player is not None:
            if player:
                return [count[0], count[2]]  #Player 1
            else:
                return [count[1], count[3]]  #Player 2
        else:
            return count

    def set_max_moves_done(self, done=True):
        self.max_moves_done = done

    def num_attacks(self):
        """
        Returns: total number of pieces to which this player is attacking
        """
        piece_locations = self.board.get_piece_locations()

        capture_moves = reduce(lambda x, y: x + y, list(map(self.board.get_capture_moves, piece_locations)), [])
        num_pieces_in_attack = 0

        pieces_in_attack = set()
        for move in capture_moves:
            for i, loc in enumerate(move):
                if (i+1) < len(move):
                    loc_2 = move[i+1]
                    pieces_in_attack.add(( (loc_2[0] + loc[0]) / 2, (loc_2[1] + loc[1]) / 2 + loc[0] % 2,))

        num_pieces_in_attack = len(pieces_in_attack)
        return num_pieces_in_attack

class ClassicGameRules:
    """
    This class is used to control the flow of game.
    The only control right now is whether to show game board at every step or not.
    """

    def __init__(self, max_moves=200, board=None):
        self.max_moves = max_moves
        self.quiet = False

    def prev_game(self, first_agent, second_agent, first_agent_turn, quiet=False, board=None):
        init_state = GameState(the_player_turn=first_agent_turn)
        
        init_state.board.spots = board
        self.quiet = quiet
        game = Game(first_agent, second_agent, init_state, self)

        return game
    
    def new_game(self, first_agent, second_agent, first_agent_turn, quiet=False):
        init_state = GameState(the_player_turn=first_agent_turn)
        print('INITIAL STATE')
        print(init_state.board.spots)
        self.quiet = quiet
        game = Game(first_agent, second_agent, init_state, self)

        return game


def load_agent(agent_type, agent_learn, weights=None, depth=3):
    """
    agent_type: type of agent, e.g. k, ab, rl

    Returns: instance of the respective agent
    """

    if agent_type == 'k':
        return KeyBoardAgent()
    elif agent_type == 'ab':
        return AlphaBetaAgent(depth=depth)
    elif agent_type == 'ql':
        is_learning_agent = True if agent_learn else False
        return QLearningAgent(is_learning_agent=is_learning_agent, weights=weights)
    elif agent_type == 'sl':
        is_learning_agent = True if agent_learn else False
        return SarsaLearningAgent(is_learning_agent=is_learning_agent, weights=weights)
    elif agent_type == 'ssl':
        is_learning_agent = True if agent_learn else False
        return SarsaSoftmaxAgent(is_learning_agent=is_learning_agent, weights=weights)
    else:
        raise Exception('Invalid agent ' + str(agent_type))


def default(str):
    return str + ' [Default: %default]'


def read_command(argv):
    """
    Processes the command used to run pacman from the command line.
    """

    from optparse import OptionParser

    usage_str = """
    USAGE:      python checkers.py <options>
    EXAMPLES:   (1) python checkers.py
                    - starts a two player game
    """
    parser = OptionParser(usage_str)

    parser.add_option('-n', '--numGames', dest='num_games', type='int',
                      help=default('the number of GAMES to play'), metavar='GAMES', default=1)

    # k for keyboard agent
    # ab for alphabeta agent
    # rl for reinforcement learning agent
    parser.add_option('-f', '--agentFirstType', dest='first_agent', type='string',
                      help=default('the first agent of game'), default='k')

    parser.add_option('-l', '--agentFirstLearn', dest='first_agent_learn', type='int',
                      help=default('the first agent of game is learning ' +
                        '(only applicable for learning agents)'), default=1)


    parser.add_option('-s', '--agentSecondType', dest='second_agent', type='string',
                      help=default('the second agent of game'), default='k')

    parser.add_option('-d', '--agentsecondLearn', dest='second_agent_learn', type='int',
                      help=default('the second agent of game is learning ' +
                        '(only applicable for learning agents)'), default=1)


    parser.add_option('-t', '--turn', dest='turn', type='int', 
                      help=default('which agent should take first turn'), default=1)

    parser.add_option('-r', '--updateParam', dest='update_param', type='int',
                      help=default('update learning parameters as time passes'), default=0)

    parser.add_option('-q', '--quiet', dest='quiet', type='int', 
                      help=default('to be quiet or not'), default=0)

    parser.add_option('-x', '--firstAgentSave', dest='first_save', type='string',
                      help=default('file to save for the first agent (used only ' +
                        'if this agent is a learning agent)'), default='./data/first_save')

    parser.add_option('-y', '--secondAgentSave', dest='second_save', type='string',
                      help=default('file to save for the second agent (used only ' +
                        'if this agent is a learning agent)'), default='./data/second_save')

    parser.add_option('-z', '--firstAgentWeights', dest='first_weights', type='string',
                      help=default('file to save weights for the first agent (used only ' +
                        'if this agent is a learning agent)'), default='./data/first_weights')

    parser.add_option('-w', '--secondAgentWeights', dest='second_weights', type='string',
                      help=default('file to save weights for the second agent (used only ' +
                        'if this agent is a learning agent)'), default='./data/second_weights')

    parser.add_option('-u', '--firstResult', dest='first_results', type='string',
                      help=default('file to save results for the first agent (used only ' +
                        'if this agent is a learning agent)'), default='./data/first_results')

    parser.add_option('-v', '--secondResult', dest='second_results', type='string',
                      help=default('file to save results for the second agent (used only ' +
                        'if this agent is a learning agent)'), default='./data/second_results')

    parser.add_option('-g', '--firstMResult', dest='first_m_results', type='string',
                      help=default('file to save num moves for the first agent (used only ' +
                        'if this agent is a learning agent)'), default='./data/first_m_results')

    parser.add_option('-i', '--secondMResult', dest='second_m_results', type='string',
                      help=default('file to save num moves for the second agent (used only ' +
                        'if this agent is a learning agent)'), default='./data/second_m_results')


    parser.add_option('-p', '--playSelf', dest='play_against_self', type='int',
                      help=default('whether first agent is to play agains itself (only' +
                        'for rl agents)'), default=0)


    options, garbage = parser.parse_args(argv)

    if len(garbage) > 0:
        raise Exception('Command line input not understood ' + str(garbage))

    args = dict()

    args['num_games'] = options.num_games

    first_weights = load_weights(options.first_weights)
    args['first_agent'] = load_agent(options.first_agent, options.first_agent_learn, first_weights)

    second_weights = load_weights(options.second_weights)
    args['second_agent'] = load_agent(options.second_agent, options.second_agent_learn, second_weights)

    args['first_agent_turn'] = options.turn == 1

    args['update_param'] = options.update_param

    args['quiet'] = True if options.quiet else False

    args['first_file_name'] = options.first_save
    args['second_file_name'] = options.second_save

    args['first_weights_file_name'] = options.first_weights
    args['second_weights_file_name'] = options.second_weights

    args['first_result_file_name'] = options.first_results
    args['second_result_file_name'] = options.second_results

    args['first_m_result_file_name'] = options.first_m_results
    args['second_m_result_file_name'] = options.second_m_results


    args['play_against_self'] = options.play_against_self == 1

    return args


def run_test(rules, first_agent, second_agent, first_agent_turn, quiet=True):
    game = rules.new_game(first_agent, second_agent, first_agent_turn, quiet=True)
    num_moves, game_state = game.run()
    return num_moves, game_state


def multiprocess(rules, first_agent, second_agent, first_agent_turn, quiet=True):
    results = []

    result_f = [[], []]
    result_s = [[], []]

    pool = Pool(4)
    kwds = {'quiet': quiet}
    for i in range(TEST_GAMES):
        results.append(pool.apply_async(run_test, [rules, first_agent, second_agent, 
            first_agent_turn], kwds))

    pool.close()
    pool.join()

    for result in results:
        val = result.get()
        num_moves, game_state = val[0], val[1]

        if first_agent.has_been_learning_agent:
            if game_state.max_moves_done:
                result_f[0].append(0.5)
            else:
                result_f[0].append(1 if game_state.is_first_agent_win() else 0)

            result_f[1].append(num_moves)

        if second_agent.has_been_learning_agent:
            if game_state.max_moves_done:
                result_s[0].append(0.5)
            else:
                result_s[0].append(1 if game_state.is_second_agent_win() else 0)

            result_s[1].append(num_moves)

    return result_f, result_s

def format_moves(move):
    M = ""
    for i in range(2):
        y = move[i][1]
        x = move[i][0]
        y = 3 - y
        y = 2*y + (x + 1)%2
        x = 7 - x

        move[i][1] = y
        move[i][0] = x

    M = str(move[0][0]) + " " + str(move[0][1]) + " " + str(move[1][0]) + " " + str(move[1][1])
    return M
        
def play_rl(first_agent, second_agent, first_agent_turn, num_games, update_param=0, quiet=False, 
                first_file_name="./data/first_save", second_file_name="./data/second_save", 
                first_weights_file_name="./data/first_weights", 
                second_weights_file_name="./data/second_weights",
                first_result_file_name="./data/first_results",
                second_result_file_name="./data/second_results", 
                first_m_result_file_name="./data/first_m_results",
                second_m_result_file_name="./data/second_m_results", 
                play_against_self=False, board=None):
    try:
        write_str = "num_moves,win,reward,max_q_value\n"


        # learn weights
        # save weights
        # test using weights
        # change agent

        print('starting game', 0)
        for i in range(num_games):

            if (i+1) % NOTIFY_FREQ == 0:
                print('Starting game', (i+1))

            rules = ClassicGameRules()


            game = rules.prev_game(first_agent, second_agent, first_agent_turn, quiet=quiet, board=board)

            move = game.run_move()

            print('THIS IS THE MOVE')
            print(move)
            print('RL agent move: ')
            formated_moves =  format_moves(move)
            print(formated_moves)
            return formated_moves

            

            

            





    except Exception as e:
        print(sys.exc_info()[0])
        traceback.print_tb(e.__traceback__)
    

def flipBoard(board):
    for i in range(8):
            for j in range(4):
                board[i][j], board[i][7 - j] = board[i][7 - j], board[i][j]

    for i in range(4):
        board[i], board[7 - i] = board[7 - i], board[i]
    return board

def clean_board(a):
    board = [[0 for x in range(8)] for y in range(8)]
    for i in range(8):
        for j in range(8):
            board[i][j] = a[i][j]
    return board

def swap_color(x):
    piece = {1: 2, 2: 1, 3: 4, 4: 3}
    return piece[x]

def getInternalRepresentation(a):
    board = []
    skip = 0
    
    for i in range(8):
        row = []
        for j in range(8):
            if(skip % 2 == 0):
                if(int(a[i][j]) != 0):
                    row.append( swap_color(int(a[i][j])))
                else:
                    row.append(0)
            skip+=1
        skip+=1
        if(len(row) != 0):                    ## check this
            board.append(row)
    # print(board)
        
    return board


def getRLMove(board):
    board = clean_board(board)
    board = flipBoard(board)
    
    board = getInternalRepresentation(board)
    #return board
    args = read_command(['-f', 'sl', '-s', 'k', '-z', './s_ab_3/first_weights', '-l', '0', '-t', '1'])
    args['board'] = board
    move = play_rl(**args)

    return move
