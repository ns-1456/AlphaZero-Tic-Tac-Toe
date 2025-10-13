import numpy as np
import torch
from abc import ABC, abstractmethod

class Game(ABC):
    """Abstract base class for games compatible with AlphaZero."""
    
    def __init__(self):
        self.board_size = None
        self.action_size = None
        self.players = None
    
    @abstractmethod
    def get_initial_state(self):
        """Return the initial state of the game."""
        pass
    
    @abstractmethod
    def get_next_state(self, state, action, player):
        """Return the new state after taking an action."""
        pass
    
    @abstractmethod
    def get_valid_moves(self, state):
        """Return a mask of valid moves (1 for valid, 0 for invalid)."""
        pass
    
    @abstractmethod
    def check_win(self, state, action):
        """Check if the last action resulted in a win."""
        pass
    
    @abstractmethod
    def get_value_and_terminated(self, state, action):
        """Return (value, is_terminal) for the current state."""
        pass
    
    @abstractmethod
    def get_encoded_state(self, state):
        """Return the state encoded as a tensor for the neural network."""
        pass
    
    @abstractmethod
    def get_opponent(self, player):
        """Return the opponent player."""
        pass
    
    @abstractmethod
    def change_perspective(self, state, player):
        """Change the perspective of the state to the given player."""
        pass

class TicTacToe(Game):
    """Tic-Tac-Toe game implementation."""
    
    def __init__(self):
        super().__init__()
        self.board_size = (3, 3)
        self.action_size = 9  # 3x3 = 9 positions
        self.players = [1, -1]  # X=1, O=-1
        
    def get_initial_state(self):
        """Return empty 3x3 board."""
        return np.zeros((3, 3), dtype=np.int8)
    
    def get_next_state(self, state, action, player):
        """Return new state after placing piece."""
        new_state = state.copy()
        row, col = action // 3, action % 3
        new_state[row, col] = player
        return new_state
    
    def get_valid_moves(self, state):
        """Return mask of valid moves (empty positions)."""
        valid_moves = np.zeros(self.action_size, dtype=np.int8)
        for i in range(self.action_size):
            row, col = i // 3, i % 3
            if state[row, col] == 0:
                valid_moves[i] = 1
        return valid_moves
    
    def check_win(self, state, action):
        """Check if the last action resulted in a win."""
        if action is None:
            return False
            
        row, col = action // 3, action % 3
        player = state[row, col]
        
        # Check row
        if np.all(state[row, :] == player):
            return True
        
        # Check column
        if np.all(state[:, col] == player):
            return True
        
        # Check diagonals
        if row == col and np.all(np.diag(state) == player):
            return True
        
        if row + col == 2 and np.all(np.diag(np.fliplr(state)) == player):
            return True
        
        return False
    
    def get_value_and_terminated(self, state, action):
        """Return (value, is_terminal) for the current state."""
        if action is None:
            return 0, False
            
        # Check if last move won
        if self.check_win(state, action):
            return 1, True
        
        # Check if board is full (draw)
        if np.all(state != 0):
            return 0, True
        
        return 0, False
    
    def get_encoded_state(self, state):
        """Return state encoded as tensor for neural network.
        
        Returns 3x3x3 tensor:
        - Plane 0: Current player's pieces (1 where they have pieces)
        - Plane 1: Opponent's pieces (1 where opponent has pieces)  
        - Plane 2: Turn indicator (all 1s for current player's turn)
        """
        encoded = np.zeros((3, 3, 3), dtype=np.float32)
        
        # Current player is always 1, opponent is -1
        current_player = 1
        opponent = -1
        
        # Fill planes
        encoded[0] = (state == current_player).astype(np.float32)  # Current player pieces
        encoded[1] = (state == opponent).astype(np.float32)       # Opponent pieces
        encoded[2] = np.ones((3, 3), dtype=np.float32)            # Turn indicator
        
        return torch.tensor(encoded, dtype=torch.float32)
    
    def get_opponent(self, player):
        """Return the opponent player."""
        return -player
    
    def change_perspective(self, state, player):
        """Change the perspective of the state to the given player."""
        if player == 1:
            return state
        else:
            # Flip the board for player -1's perspective
            return -state
    
    def get_winning_moves(self, state, player):
        """Get moves that would win for the given player."""
        winning_moves = []
        for i in range(self.action_size):
            row, col = i // 3, i % 3
            if state[row, col] == 0:  # Empty position
                # Try the move
                test_state = state.copy()
                test_state[row, col] = player
                if self.check_win(test_state, i):
                    winning_moves.append(i)
        return winning_moves
    
    def get_blocking_moves(self, state, player):
        """Get moves that would block opponent from winning."""
        opponent = self.get_opponent(player)
        return self.get_winning_moves(state, opponent)
    
    def display(self, state):
        """Display the current board state."""
        symbols = {1: 'X', -1: 'O', 0: '.'}
        print("  0 1 2")
        for i in range(3):
            print(f"{i} {symbols[state[i,0]]} {symbols[state[i,1]]} {symbols[state[i,2]]}")
        print()
