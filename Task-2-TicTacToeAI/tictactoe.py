# Task 2 - Tic Tac Toe AI (CodSoft Internship)

board = [" " for _ in range(9)]

def print_board():
    print()
    for i in range(0, 9, 3):
        print(board[i] + " | " + board[i+1] + " | " + board[i+2])
        if i < 6:
            print("--+---+--")
    print()

def check_winner(player):
    win_conditions = [
        (0,1,2), (3,4,5), (6,7,8),
        (0,3,6), (1,4,7), (2,5,8),
        (0,4,8), (2,4,6)
    ]
    for a,b,c in win_conditions:
        if board[a] == board[b] == board[c] == player:
            return True
    return False

def is_full():
    return " " not in board

def player_move():
    while True:
        move = int(input("Enter position (1-9): ")) - 1
        if board[move] == " ":
            board[move] = "X"
            break
        else:
            print("Invalid move!")

def ai_move():
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            if check_winner("O"):
                return
            board[i] = " "
    for i in range(9):
        if board[i] == " ":
            board[i] = "X"
            if check_winner("X"):
                board[i] = "O"
                return
            board[i] = " "
    for i in range(9):
        if board[i] == " ":
            board[i] = "O"
            return

# Game loop
print("Welcome to Tic Tac Toe AI!")
print_board()

while True:
    player_move()
    print_board()

    if check_winner("X"):
        print("You Win!")
        break

    if is_full():
        print("Draw!")
        break

    ai_move()
    print_board()

    if check_winner("O"):
        print("AI Wins!")
        break

    if is_full():
        print("Draw!")
        break
