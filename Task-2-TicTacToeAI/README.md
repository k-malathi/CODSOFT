# Task 2: Tic-Tac-Toe AI

## CODSOFT Artificial Intelligence Internship

###  Overview
This project implements an **unbeatable Tic-Tac-Toe AI agent** that plays
against a human player in the terminal. The AI uses the **Minimax
algorithm with Alpha-Beta Pruning** to search the game tree and always
choose the optimal move — meaning a perfect human opponent can, at best,
force a draw.

This task demonstrates core concepts in:
- Game theory (zero-sum, perfect-information games)
- Adversarial search (Minimax)
- Search-space optimization (Alpha-Beta Pruning)

---

###  How the AI Works

#### Minimax Algorithm
Minimax explores every possible sequence of future moves from the
current board state, assuming:
- The **AI ('O')** is the *maximizing* player — it picks moves that lead
  to the best possible outcome for itself.
- The **human ('X')** is the *minimizing* player — Minimax assumes the
  opponent will always play optimally against the AI.

Each terminal (game-over) board state is scored:
| Outcome              | Score            |
|-----------------------|------------------|
| AI wins                | `10 - depth`     |
| Human wins             | `depth - 10`     |
| Draw                    | `0`              |

Subtracting/adding `depth` makes the AI prefer **faster wins** and
**slower losses**, so it plays more human-like and decisive rather than
delaying an inevitable win.

#### Alpha-Beta Pruning
Alpha-Beta Pruning is an optimization on top of Minimax that skips
("prunes") branches of the game tree that can't possibly influence the
final decision, based on two bounds:
- **Alpha** — the best score the maximizer (AI) can guarantee so far
- **Beta** — the best score the minimizer (human) can guarantee so far

Whenever `beta <= alpha`, the remaining branches at that node are
skipped, since no rational opponent would ever let that branch happen.
This dramatically reduces the number of nodes evaluated without
changing the final result — the AI is still perfectly optimal, just
faster.

---

###  Features
- **Unbeatable Hard mode** — true Minimax + Alpha-Beta Pruning
- **Easy / Medium difficulty** — AI occasionally plays a random
  (suboptimal) move so beginners have a chance to win
- **Choice of turn order** — play first (X) or let the AI go first
- **Clean terminal board rendering** with a position guide (0–8)
- **Win / Draw detection**
- **Optional AI vs AI demo** — watch two perfect AIs always draw,
  proving the algorithm's correctness

---

###  How to Run

**Requirements:** Python 3.x (standard library only — no external dependencies)

```bash
python tic_tac_toe_ai.py
```

You'll be prompted to:
1. Choose a difficulty (Easy / Medium / Hard)
2. Choose whether to go first or second
3. Enter moves as numbers **0–8** corresponding to the board position:

```
 0 | 1 | 2
---+---+---
 3 | 4 | 5
---+---+---
 6 | 7 | 8
```

To watch two AIs play a perfect game against each other instead, open
`tic_tac_toe_ai.py` and uncomment this line at the bottom:
```python
run_ai_vs_ai_demo()
```

---

###  Files
```
tic_tac_toe_ai.py   # Full game + Minimax AI implementation
README.md           # Project documentation (this file)
```

---

###  Example Interaction
```
Select AI difficulty:
  1. Easy
  2. Medium
  3. Hard (Unbeatable)
Enter choice (1-3): 3

Do you want to go first or second?
  1. First (you are X)
  2. Second (AI goes first, you are X)
Enter choice (1-2): 1

Enter your move (0-8): 0
AI is thinking...
AI chose position 4
...
It's a draw! 
```

On **Hard** difficulty, the best a human can ever achieve against this
AI is a draw — it never loses.

---

###  Possible Extensions
- Add a GUI using Tkinter or Pygame
- Extend to a larger board (e.g. 4x4, connect-4 style) with adjustable
  win-length
- Add a scoreboard to track wins/losses/draws across multiple rounds
- Implement Minimax with move ordering or transposition tables for
  larger game trees

---

###  Internship Info
This task was completed as part of the **CODSOFT Artificial Intelligence
Internship**. Repository maintained under: `CODSOFT`

#codsoft #internship #artificialintelligence #gametheory
