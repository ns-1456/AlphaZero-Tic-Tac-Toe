import torch
import numpy as np
from model import TicTacToeNet
from game import TicTacToe
from mcts import MCTS

def demo_model():
    """Demonstrate the AlphaZero Tic-Tac-Toe model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model and game
    model = TicTacToeNet().to(device)
    model.eval()
    
    game = TicTacToe()
    mcts = MCTS(model, game, num_simulations=50, device=device)
    
    # Start a game
    state = game.get_initial_state()
    print("Starting position:")
    game.display(state)
    print()
    
    move_count = 0
    current_player = 1
    
    while not game.get_value_and_terminated(state, None)[1] and move_count < 9:
        print(f"Move {move_count + 1}: {'X' if current_player == 1 else 'O'} to move")
        
        # Get model's move
        policy = mcts.search(state)
        valid_moves = game.get_valid_moves(state)
        masked_policy = policy * valid_moves
        action = masked_policy.argmax().item()
        
        row, col = action // 3, action % 3
        print(f"Model plays: {action} (row {row}, col {col})")
        
        # Show move probabilities
        print("Move probabilities:")
        for i in range(9):
            row_i, col_i = i // 3, i % 3
            prob = policy[i].item()
            if valid_moves[i] == 1:
                print(f"  {row_i},{col_i}: {prob:.3f}")
        
        state = game.get_next_state(state, action, current_player)
        current_player = game.get_opponent(current_player)
        move_count += 1
        
        game.display(state)
        print()
    
    # Determine result
    game_value, _ = game.get_value_and_terminated(state, None)
    if game_value == 1:
        print("X wins!")
    elif game_value == -1:
        print("O wins!")
    else:
        print("It's a draw!")
    
    print("Demo completed!")

def demo_training_progress():
    """Demonstrate how the model improves during training."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    game = TicTacToe()
    
    # Test untrained model
    print("Testing untrained model...")
    model = TicTacToeNet().to(device)
    model.eval()
    
    mcts = MCTS(model, game, num_simulations=50, device=device)
    state = game.get_initial_state()
    
    print("Untrained model's move probabilities:")
    policy = mcts.search(state)
    for i in range(9):
        row, col = i // 3, i % 3
        prob = policy[i].item()
        print(f"  {row},{col}: {prob:.3f}")
    
    print("\nUntrained model plays:")
    game.display(state)
    
    # Simulate a few moves
    for move in range(3):
        valid_moves = game.get_valid_moves(state)
        masked_policy = policy * valid_moves
        action = masked_policy.argmax().item()
        
        row, col = action // 3, action % 3
        print(f"Move {move + 1}: {action} (row {row}, col {col})")
        
        state = game.get_next_state(state, action, 1)
        game.display(state)
        
        if game.get_value_and_terminated(state, None)[1]:
            break
        
        # Get next policy
        policy = mcts.search(state)
    
    print("Demo completed!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == 'training':
        demo_training_progress()
    else:
        demo_model()