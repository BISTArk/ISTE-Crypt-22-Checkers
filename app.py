from flask import Flask,request
from model import finMove
from RL_module import make_board
from RL_module.flask_rl import getRLMove
app = Flask(__name__)

rl_model = 1

@app.route('/',methods = ["GET","POST"])
def haha():
    print(request.method)

    if request.method=="POST":
        board = make_board(str(request.data.decode("utf-8")))
    
        MoveAlphaBeta = finMove(board)
        MoveRL = getRLMove(board)
        
        print()

        print('Alpha Beta agent move:')
        print(MoveAlphaBeta)
        if(rl_model):
            return MoveRL
        else:
            return MoveAlphaBeta

    return "dont get daa"