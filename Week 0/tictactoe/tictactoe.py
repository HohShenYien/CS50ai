"""
Tic Tac Toe Player
"""
import copy
import math

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """
    Returns starting state of the board.
    """
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns player who has the next turn on a board.
    """
    # First player is X
    # So I check how many non-empty cells are there to determine the number of rounds
    rounds = 0
    for row in board:
        for cell in row:
            if cell != EMPTY:
                rounds += 1

    # Even number round, 0, 2, 4... is first player(X)
    if rounds % 2 == 0:
        return X

    return O


def actions(board):
    """
    Returns set of all possible actions (i, j) available on the board.
    """

    # Place available actions into a list
    actions = []
    for i in range(3):
        for j in range(3):
            # If empty, means an available move
            if board[i][j] == EMPTY:
                actions.append((i,j))
    return actions


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    """
    # Do not alter the original board
    # So I create a copy of it and return with the move
    board_copy = copy.deepcopy(board)

    # Check if it's a valid move
    # First see if it is within the size
    try:
        if board_copy[action[0]][action[1]]:
            raise Exception("in", board, "\n", action, "is not a valid move")

    except:
        print(action)
        raise Exception("Not a valid move")


    # Find out which is the current player so that I can place the sign on the board
    cur_player = player(board_copy)
    board_copy[action[0]][action[1]] = cur_player
    return board_copy


def winner(board):
    """
    Returns the winner of the game, if there is one.
    """
    # Straight forward to find winner

    if not terminal(board):
        return None

    winner = utility(board)
    if winner == 1:
        return X

    elif winner == -1:
        return O

    else:
        return None


def terminal(board):
    """
    Returns True if game is over, False otherwise.
    """
    # Check if a winner is present, if yes then game ends
    if utility(board):
        return True

    # Check if the board still have empty cell
    for row in board:
        for cell in row:
            if cell is EMPTY:
                return False

    # If all cells are filled
    return True


def utility(board):
    """
    Returns 1 if X has won the game, -1 if O has won, 0 otherwise.
    """
    # A dictionary so that I can return the winner easily
    players = {"X": 1, "O": -1}

    # Must ensure those are not empty or 'None'
    # Check for row first
    for row in board:
        if row[0]:
            if row[0] == row[1] and row[1] == row[2]:
                    return players[row[0]]


    # Check for each column
    for j in range(3):
        if board[0][j]:
            if board[0][j] == board[1][j] and board[1][j] == board[2][j]:
                return players[board[0][j]]

    # Check for diagonal
    if board[1][1]:
        if board[0][0] == board[1][1] and board[1][1] == board[2][2]:
            return players[board[0][0]]

        if board[0][2] == board[1][1] and board[1][1] == board[2][0]:
            return players[board[0][2]]

    # None of them win
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    """
    # A recursive function, just as told by Brian from lecture video

    # Checks the current player
    cur_player = player(board)

    #Check for available moves
    available_moves = actions(board)

    # Base case, that is, when it is last move for X
    if len(available_moves) == 1:
        return available_moves.pop()
    # Best move for empty board
    elif len(available_moves) == 9:
        return (0, 0)

    # Now execute each find_max and find_min function
    if cur_player == X:
        # Check for minimum values after each step is made
        cur_max = find_min(result(board, available_moves[0]))
        cur_move = available_moves[0]

        for action in available_moves[1:]:
            value = find_min(result(board, action), cur_max)
            if value is None:
                continue
            if value > cur_max:
                cur_max = value
                cur_move = action

        return cur_move

    # Similarly for O
    elif cur_player == O:
        cur_min = find_max(result(board, available_moves[0]))
        cur_move = available_moves[0]
        for action in available_moves[1:]:
            value = find_max(result(board, action), cur_min)
            if value is None:
                continue
            if value < cur_min:
                cur_min = value
                cur_move = action
        return cur_move


# prev_max is the maximum of actions in the same level
'''
Something like
    find_min
  /       \
 5         find_max

 and the prev_max will be 5 in this case
 so any number(say 6) larger than 5 will straight away cancel the find_max function
 because find_max will pick 6 or larger number,
 and subsequently find_min will definitely pick 5
 So this can improve the performance a bit
'''
# for alpha-beta pruning
# prev_max is infinity in default

# Find the maximum value of this step
def find_max(board, prev_max = float('inf')):
    # If completed, straight away get the result
    if terminal(board):
        return utility(board)

    cur_max = float('-inf')
    for action in actions(board):
        if cur_max >= prev_max:
            # Do not need to consider this action anymore and skip to next
            return None
        value = find_min(result(board, action), cur_max)
        # Making sure if the next action is necessary to consider or not
        if value is not None:
            cur_max = max(cur_max, value)

    return cur_max

# Find the minimum at this step
def find_min(board, prev_min = float('-inf')):
    #Similarly for find_min
    if terminal(board):
        return utility(board)

    cur_min = float('inf')
    for action in actions(board):
        if cur_min <= prev_min:
            return None
        value = find_max(result(board, action), cur_min)

        if value is not None:
            cur_min = min(cur_min, value)
    return cur_min


