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

from util import open_file, load_weights

from game import *
from agents import *

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
            print(format_moves(move))
            return format_moves(move)

            

            

            





    except Exception as e:
        print(sys.exc_info()[0])
        traceback.print_tb(e.__traceback__)
    

def flipBoard(board):
    for i in range(8):
            for j in range(4):
                board[i][j], board[i][7 - j] = board[i][7 - j], board[i][j]
    return board

def clean_board(a):
    board = [[0 for x in range(8)] for y in range(8)]
    for i in range(8):
        for j in range(8):
            board[i][j] = a[i][j]
    return board

def getInternalRepresentation(a):
    board = [[]]
    for i in a:
        for j in i:
            if(j != 0):
                board.append(j)
    return board


def getRLMove(board):
    board = clean_board(board)
    board = flipBoard(board)
    board = getInternalRepresentation(board)
    args = read_command(['-f', 'sl', '-s', 'k', '-z', './s_ab_3/first_weights', '-l', '0', '-t', '1'])
    args['board'] = board
    move = play_rl(**args)

    return move




start_time = time.time()
args = read_command(['-f', 'sl', '-s', 'k', '-z', './s_ab_3/first_weights', '-l', '0', '-t', '1'])
# run_games(**args)

args['board'] = [[j, j, j, j] for j in [3, 1, 1, 0, 
                                                0, 2, 2, 2]]
print(args)
play_rl(**args)
print(time.time() - start_time)