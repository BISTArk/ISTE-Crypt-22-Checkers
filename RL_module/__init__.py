def make_board(data):
    temp = data.split("\n",8)
    temp = [i.split(" ",8) for i in temp]
    print("Temp = ", temp)
    
    return temp

