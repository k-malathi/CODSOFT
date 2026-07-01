"""
CODSOFT - Artificial Intelligence Internship
Task 2: Tic-Tac-Toe AI

An unbeatable Tic-Tac-Toe AI agent that plays against a human player,
using the Minimax algorithm with Alpha-Beta Pruning for efficient
game-tree search.

Features:
- Unbeatable AI (Minimax + Alpha-Beta Pruning)
- Difficulty levels (Easy / Medium / Hard) for a more human-friendly experience
- Choice of playing first or second
- Clean board rendering in the terminal
- Win / Draw detection
"""

import math
import random


class TicTacToe:
    """Encapsulates the Tic-Tac-Toe board state and game rules."""

    def __init__(self):
        self.board = [" "] * 9
        self.human = "X"
        self.ai = "O"

    def print_board(self):
        b = self.board
        print()
        print(f" {b[0]} | {b[1]} | {b[2]} ")
        print("---+---+---")
        print(f" {b[3]} | {b[4]} | {b[5]} ")
        print("---+---+---")
        print(f" {b[6]} | {b[7]} | {b[8]} ")
        print()

    def available_moves(self):
        return [i for i, spot in enumerate(self.board) if spot == " "]

    def make_move(self, position, player):
        if self.board[position] == " ":
            self.board[position] = player
            return True
        return False

    def undo_move(self, position):
        self.board[position] = " "

    WIN_COMBOS = [
        (0, 1, 2), (3, 4, 5), (6, 7, 8),   
        (0, 3, 6), (1, 4, 7), (2, 5, 8),  
        (0, 4, 8), (2, 4, 6),             
    ]

    def winner(self):
        """Return 'X', 'O', 'Draw', or None (game still in progress)."""
        for a, b, c in self.WIN_COMBOS:
            if self.board[a] != " " and self.board[a] == self.board[b] == self.board[c]:
                return self.board[a]
        if " " not in self.board:
            return "Draw"
        return None

    def is_game_over(self):
        return self.winner() is not None


class TicTacToeAI:
    """
    AI agent that chooses moves using Minimax with Alpha-Beta Pruning.
    Optionally supports 'easy' and 'medium' difficulty by occasionally
    picking a random or suboptimal move instead of the perfect one.
    """

    def __init__(self, game: TicTacToe, difficulty="hard"):
        self.game = game
        self.difficulty = difficulty  

    def minimax(self, depth, is_maximizing, alpha, beta):
        """
        Core Minimax algorithm with Alpha-Beta Pruning.

        - Maximizing player = AI ('O') -> tries to maximize the score
        - Minimizing player = Human ('X') -> tries to minimize the score
        - Alpha = best score maximizer can guarantee so far
        - Beta  = best score minimizer can guarantee so far
        - Branches are pruned (skipped) once alpha >= beta, since that
          branch can no longer affect the final decision.
        """
        result = self.game.winner()

        
        if result == self.game.ai:
            return 10 - depth
        elif result == self.game.human:
            return depth - 10
        elif result == "Draw":
            return 0

        if is_maximizing:
            best_score = -math.inf
            for move in self.game.available_moves():
                self.game.make_move(move, self.game.ai)
                score = self.minimax(depth + 1, False, alpha, beta)
                self.game.undo_move(move)
                best_score = max(best_score, score)
                alpha = max(alpha, best_score)
                if beta <= alpha:
                    break  
            return best_score
        else:
            best_score = math.inf
            for move in self.game.available_moves():
                self.game.make_move(move, self.game.human)
                score = self.minimax(depth + 1, True, alpha, beta)
                self.game.undo_move(move)
                best_score = min(best_score, score)
                beta = min(beta, best_score)
                if beta <= alpha:
                    break 
            return best_score

    def best_move(self):
        """Return the optimal move index for the AI using Minimax."""
        best_score = -math.inf
        move_choice = None
        for move in self.game.available_moves():
            self.game.make_move(move, self.game.ai)
            score = self.minimax(0, False, -math.inf, math.inf)
            self.game.undo_move(move)
            if score > best_score:
                best_score = score
                move_choice = move
        return move_choice

    def choose_move(self):
        """
        Select a move based on difficulty:
        - hard: always the optimal Minimax move (unbeatable)
        - medium: optimal move 70% of the time, random otherwise
        - easy: random move 70% of the time, optimal otherwise
        """
        available = self.game.available_moves()

        if self.difficulty == "easy":
            if random.random() < 0.7:
                return random.choice(available)
            return self.best_move()

        if self.difficulty == "medium":
            if random.random() < 0.3:
                return random.choice(available)
            return self.best_move()

       
        return self.best_move()


def get_human_move(game: TicTacToe):
    """Prompt the human player for a valid move (0-8)."""
    while True:
        try:
            move = int(input("Enter your move (0-8): "))
            if move in game.available_moves():
                return move
            print("That spot is taken or invalid. Try again.")
        except ValueError:
            print("Please enter a number between 0 and 8.")


def print_position_guide():
    print("\nPosition guide:")
    print(" 0 | 1 | 2 ")
    print("---+---+---")
    print(" 3 | 4 | 5 ")
    print("---+---+---")
    print(" 6 | 7 | 8 \n")


def choose_difficulty():
    print("Select AI difficulty:")
    print("  1. Easy")
    print("  2. Medium")
    print("  3. Hard (Unbeatable)")
    choice = input("Enter choice (1-3): ").strip()
    return {"1": "easy", "2": "medium", "3": "hard"}.get(choice, "hard")


def choose_order():
    print("\nDo you want to go first or second?")
    print("  1. First (you are X)")
    print("  2. Second (AI goes first, you are X)")
    choice = input("Enter choice (1-2): ").strip()
    return choice != "2"  # True = human goes first


def play_game():
    print("=" * 40)
    print("   TIC-TAC-TOE: Human vs Minimax AI")
    print("=" * 40)

    game = TicTacToe()
    difficulty = choose_difficulty()
    human_first = choose_order()
    ai = TicTacToeAI(game, difficulty=difficulty)

    print_position_guide()
    game.print_board()

    turn = "human" if human_first else "ai"

    while not game.is_game_over():
        if turn == "human":
            move = get_human_move(game)
            game.make_move(move, game.human)
            turn = "ai"
        else:
            print("AI is thinking...")
            move = ai.choose_move()
            game.make_move(move, game.ai)
            print(f"AI chose position {move}")
            turn = "human"

        game.print_board()

    result = game.winner()
    if result == "Draw":
        print("It's a draw! 🤝")
    elif result == game.human:
        print("Congratulations, you win! 🎉")
    else:
        print("AI wins! 🤖")


def run_ai_vs_ai_demo():
    """Optional demo: watch two Minimax AIs play a perfect (drawn) game."""
    print("\n=== AI vs AI Demo (always ends in a draw) ===")
    game = TicTacToe()
    ai_o = TicTacToeAI(game, difficulty="hard")
    game.human = "X"
    game.ai = "O"
    turn = "X"

    game.print_board()
    while not game.is_game_over():
        if turn == "X":
            game.ai, game.human = "X", "O"  
            move = ai_o.best_move()
            game.ai, game.human = "O", "X"  
            game.make_move(move, "X")
            turn = "O"
        else:
            move = ai_o.best_move()
            game.make_move(move, "O")
            turn = "X"
        game.print_board()

    print("Result:", game.winner())


if __name__ == "__main__":
    play_game()

    # Uncomment to watch two perfect AIs play each other (always a draw):
    # run_ai_vs_ai_demo()
