from flask import Flask,request
from model import make_board,finMove

app = Flask(__name__)

@app.route('/',methods = ["GET","POST"])
def haha():
    print(request.method)
    if request.method=="POST":
        board = make_board(str(request.data.decode("utf-8")))
        Move = finMove(board)
        # print(board[7][7])
        # print(request.data.decode("utf-8"))
        return Move

    return "dont get daa"