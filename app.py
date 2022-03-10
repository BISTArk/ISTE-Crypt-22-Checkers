from flask import Flask,request
from model import finMove
from RL_module import make_board
from RL_module.flask_rl import func, getRLMove
app = Flask(__name__)

@app.route('/',methods = ["GET","POST"])
def haha():
    print(request.method)
    # print(getmove())
    
    if request.method=="POST":
        board = make_board(str(request.data.decode("utf-8")))
        MoveAlphaBeta = finMove(board)
        MoveRL = getRLMove(board)
        # func()
        print()
        print('MAKE THIS CORRECT ')
        # Move_RL = getRLMove()
        # print(board[7][7])
        # print(request.data.decode("utf-8"))
        print('Alpha Beta agent move:')
        print(MoveAlphaBeta)
        return MoveRL

    return "dont get daa"