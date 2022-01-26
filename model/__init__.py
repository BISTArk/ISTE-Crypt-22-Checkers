import time
import math
from copy import deepcopy

mandatory_jumping = False

def make_board(data):
    temp = data.split("\n",8)
    temp = [i.split(" ",8) for i in temp]
    print("Temp = ", temp)

    return temp

class Node:
    def __init__(self, board, move=None, parent=None, value=None):
        self.board = board
        self.value = value
        self.move = move
        self.parent = parent

    def get_children(self, minimizing_player, mandatory_jumping):
        current_state = deepcopy(self.board)
        available_moves = []
        children_states = []
        big_letter = ""
        queen_row = 0
        if minimizing_player is True:
            available_moves = find_available_moves(current_state, mandatory_jumping)
            # print("Available moves: ", available_moves)

            big_letter = "4"
            queen_row = 7
        else:
            available_moves = find_player_available_moves(current_state, mandatory_jumping)
            # print("Available moves: ", available_moves)
            big_letter = "3"
            queen_row = 0
        for i in range(len(available_moves)):
            old_i = available_moves[i][0]
            old_j = available_moves[i][1]
            new_i = available_moves[i][2]
            new_j = available_moves[i][3]
            state = deepcopy(current_state)
            make_a_move(state, old_i, old_j, new_i, new_j, big_letter, queen_row)
            children_states.append(Node(state, [old_i, old_j, new_i, new_j]))
        return children_states

    def set_value(self, value):
        self.value = value

    def get_value(self):
        return self.value

    def get_board(self):
        return self.board

    def get_parent(self):
        return self.parent

    def set_parent(self, parent):
        self.parent = parent

def find_player_available_moves(board, mandatory_jumping):
        available_moves = []
        available_jumps = []
        for m in range(8):
            for n in range(8):
                if board[m][n] == '1':
                    if check_player_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if check_player_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if check_player_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if check_player_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
                elif board[m][n] == '3':
                    if check_player_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if check_player_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if check_player_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if check_player_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if check_player_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if check_player_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if check_player_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                    if check_player_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
        if mandatory_jumping is False:
            available_jumps.extend(available_moves)
            return available_jumps
        elif mandatory_jumping is True:
            if len(available_jumps) == 0:
                return available_moves
            else:
                return available_jumps

def check_player_jumps(board, old_i, old_j, via_i, via_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[via_i][via_j] == '0':
            return False
        if board[via_i][via_j] == '3' or board[via_i][via_j] == '1':
            return False
        if board[new_i][new_j] != '0':
            return False
        if board[old_i][old_j] == '0':
            return False
        if board[old_i][old_j] == '2' or board[old_i][old_j] == '4':
            return False
        return True

def check_player_moves(board, old_i, old_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[old_i][old_j] == '0':
            return False
        if board[new_i][new_j] != '0':
            return False
        if board[old_i][old_j] == '2' or board[old_i][old_j] == '4':
            return False
        if board[new_i][new_j] == '0':
            return True

def find_available_moves(board, mandatory_jumping):
        available_moves = []
        available_jumps = []
        for m in range(8):
            for n in range(8):
                if board[m][n][0] == "2":
                    if check_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if check_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if check_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if check_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                elif board[m][n][0] == "4":
                    if check_moves(board, m, n, m - 1, n - 1):
                        available_moves.append([m, n, m - 1, n - 1])
                    if check_moves(board, m, n, m - 1, n + 1):
                        available_moves.append([m, n, m - 1, n + 1])
                    if check_jumps(board, m, n, m - 1, n - 1, m - 2, n - 2):
                        available_jumps.append([m, n, m - 2, n - 2])
                    if check_jumps(board, m, n, m - 1, n + 1, m - 2, n + 2):
                        available_jumps.append([m, n, m - 2, n + 2])
                    if check_moves(board, m, n, m + 1, n - 1):
                        available_moves.append([m, n, m + 1, n - 1])
                    if check_jumps(board, m, n, m + 1, n - 1, m + 2, n - 2):
                        available_jumps.append([m, n, m + 2, n - 2])
                    if check_moves(board, m, n, m + 1, n + 1):
                        available_moves.append([m, n, m + 1, n + 1])
                    if check_jumps(board, m, n, m + 1, n + 1, m + 2, n + 2):
                        available_jumps.append([m, n, m + 2, n + 2])
        if mandatory_jumping is False:
            available_jumps.extend(available_moves)
            return available_jumps
        elif mandatory_jumping is True:
            if len(available_jumps) == 0:
                return available_moves
            else:
                return available_jumps

def check_moves(board, old_i, old_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[old_i][old_j] == '0':
            return False
        if board[new_i][new_j] != '0':
            return False
        if board[old_i][old_j] == '1' or board[old_i][old_j] == '3':
            return False
        if board[new_i][new_j] == '0':
            return True

def check_jumps(board, old_i, old_j, via_i, via_j, new_i, new_j):
        if new_i > 7 or new_i < 0:
            return False
        if new_j > 7 or new_j < 0:
            return False
        if board[via_i][via_j] == '0':
            return False
        if board[via_i][via_j] == '2' or board[via_i][via_j] == '4':
            return False
        if board[new_i][new_j] != '0':
            return False
        if board[old_i][old_j] == '0':
            return False
        if board[old_i][old_j] == '1' or board[old_i][old_j] == '3':
            return False
        return True

def make_a_move(board, old_i, old_j, new_i, new_j, big_letter, queen_row):
        letter = board[old_i][old_j]
        i_difference = old_i - new_i
        j_difference = old_j - new_j
        if i_difference == -2 and j_difference == 2:
            board[old_i + 1][old_j - 1] = "0"

        elif i_difference == 2 and j_difference == 2:
            board[old_i - 1][old_j - 1] = "0"

        elif i_difference == 2 and j_difference == -2:
            board[old_i - 1][old_j + 1] = "0"

        elif i_difference == -2 and j_difference == -2:
            board[old_i + 1][old_j + 1] = "0"

        if new_i == queen_row:
            letter = big_letter
        board[old_i][old_j] = "0"
        board[new_i][new_j] = letter + str(new_i) + str(new_j)

def calculate_heuristics(board):
        result = 0
        mine = 0
        opp = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] == "2" or board[i][j] == "4":
                    mine += 1

                    if board[i][j] == "2":
                        result += 5
                    if board[i][j] == "4":
                        result += 10
                    if i == 0 or j == 0 or i == 7 or j == 7:
                        result += 7
                    if i + 1 > 7 or j - 1 < 0 or i - 1 < 0 or j + 1 > 7:
                        continue
                    if (board[i + 1][j - 1] == "1" or board[i + 1][j - 1] == "3") and board[i - 1][
                        j + 1] == "0":
                        result -= 3
                    if (board[i + 1][j + 1] == "1" or board[i + 1][j + 1] == "3") and board[i - 1][j - 1] == "0":
                        result -= 3
                    if board[i - 1][j - 1] == "3" and board[i + 1][j + 1] == "0":
                        result -= 3

                    if board[i - 1][j + 1] == "3" and board[i + 1][j - 1] == "0":
                        result -= 3
                    if i + 2 > 7 or i - 2 < 0:
                        continue
                    if (board[i + 1][j - 1] == "3" or board[i + 1][j - 1] == "1") and board[i + 2][
                        j - 2] == "0":
                        result += 6
                    if i + 2 > 7 or j + 2 > 7:
                        continue
                    if (board[i + 1][j + 1] == "3" or board[i + 1][j + 1] == "1") and board[i + 2][
                        j + 2] == "0":
                        result += 6

                elif board[i][j] == "1" or board[i][j] == "3":
                    opp += 1

        return result + (mine - opp) * 1000

def handleDiff(stri):
    stri = 'HALLO ' + stri
    return stri

def minimax(board, depth, alpha, beta, maximizing_player, mandatory_jumping):
        if depth == 0:
            return calculate_heuristics(board)
        current_state = Node(deepcopy(board))
        if maximizing_player is True:
            max_eval = -math.inf
            for child in current_state.get_children(True, mandatory_jumping):
                ev = minimax(child.get_board(), depth - 1, alpha, beta, False, mandatory_jumping)
                max_eval = max(max_eval, ev)
                alpha = max(alpha, ev)
                if beta <= alpha:
                    break
            current_state.set_value(max_eval)
            return max_eval
        else:
            min_eval = math.inf
            for child in current_state.get_children(False, mandatory_jumping):
                ev = minimax(child.get_board(), depth - 1, alpha, beta, True, mandatory_jumping)
                min_eval = min(min_eval, ev)
                beta = min(beta, ev)
                if beta <= alpha:
                    break
            current_state.set_value(min_eval)
            return min_eval


def finMove(board):
        t1 = time.time()
        player_pieces = 0
        computer_pieces = 0
        for i in range(8):
            for j in range(8):
                if board[i][j] == "1" or board[i][j] == "3":
                    player_pieces += 1
                if board[i][j] == "2" or board[i][j] == "4":
                    computer_pieces += 1
        current_state = Node(deepcopy(board))

        first_computer_moves = current_state.get_children(True, mandatory_jumping)
        # print(first_computer_moves)
        if len(first_computer_moves) == 0:
            if player_pieces > computer_pieces:
                print("Computer has no available moves left, and you have more pieces left.\nYOU WIN!")
                return handleDiff("Computer has no available moves left, and you have more pieces left.\nYOU WIN!")
            else:
                print("Computer has no available moves left.\nGAME ENDED!")
                return handleDiff("Computer has no available moves left.\nGAME ENDED!")
        dict = {}
        for i in range(len(first_computer_moves)):
            child = first_computer_moves[i]
            value = minimax(child.get_board(), 4, -math.inf, math.inf, False, mandatory_jumping)
            dict[value] = child
        if len(dict.keys()) == 0:
            print("Computer has cornered itself.\nYOU WIN!")
            return handleDiff("Computer has cornered itself.\nYOU WIN!")
        move = dict[max(dict)].move
        t2 = time.time()
        diff = t2 - t1
        print("Computer has moved (" + str(move[0]) + "," + str(move[1]) + ") to (" + str(move[2]) + "," + str(
            move[3]) + ").")
        print("It took him " + str(diff) + " seconds.")

        return(str(move[0])+" "+str(move[1])+" "+str(move[2])+" "+str(move[3]))