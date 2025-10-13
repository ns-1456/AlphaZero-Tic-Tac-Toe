import torch
import numpy as np
import random
from model import TicTacToeNet
from game import TicTacToe
from mcts import MCTS

def minimax(game, state, player, depth=0, max_depth=9):
    """Minimax algorithm with alpha-beta pruning for optimal play."""
    game_value, is_terminal = game.get_value_and_terminated(state, None)
    
    if is_terminal or depth >= max_depth:
        return game_value, None
    
    valid_moves = game.get_valid_moves(state)
    best_value = float('-inf') if player == 1 else float('inf')
    best_action = None
    
    for action in range(game.action_size):
        if valid_moves[action] == 1:  # Valid move
            next_state = game.get_next_state(state, action, player)
            value, _ = minimax(game, next_state, game.get_opponent(player), depth + 1, max_depth)
            
            if player == 1:  # Maximizing player
                if value > best_value:
                    best_value = value
                    best_action = action
            else:  # Minimizing player
                if value < best_value:
                    best_value = value
                    best_action = action
    
    return best_value, best_action

def play_game(model, game, device, opponent='random', verbose=True):
    """Play a game between the model and an opponent."""
    state = game.get_initial_state()
    mcts = MCTS(model, game, num_simulations=50, device=device)
    current_player = 1
    move_count = 0
    
    if verbose:
        print("Starting game: Model (X) vs " + opponent.title() + " (O)")
        game.display(state)
    
    while not game.get_value_and_terminated(state, None)[1] and move_count < 9:
        if current_player == 1:
            # Model plays
            policy = mcts.search(state)
            valid_moves = game.get_valid_moves(state)
            masked_policy = policy * valid_moves
            action = masked_policy.argmax().item()
            
            if verbose:
                print(f"Model plays: {action} (row {action//3}, col {action%3})")
        else:
            # Opponent plays
            if opponent == 'random':
                valid_moves = game.get_valid_moves(state)
                valid_actions = np.where(valid_moves == 1)[0]
                action = np.random.choice(valid_actions)
            elif opponent == 'optimal':
                _, action = minimax(game, state, current_player)
                if action is None:
                    valid_moves = game.get_valid_moves(state)
                    valid_actions = np.where(valid_moves == 1)[0]
                    if len(valid_actions) > 0:
                        action = np.random.choice(valid_actions)
                    else:
                        break
            
            if verbose:
                print(f"{opponent.title()} plays: {action} (row {action//3}, col {action%3})")
        
        state = game.get_next_state(state, action, current_player)
        current_player = game.get_opponent(current_player)
        move_count += 1
        
        if verbose:
            game.display(state)
    
    # Determine result
    game_value, _ = game.get_value_and_terminated(state, None)
    if game_value == 1:
        result = "Model wins!"
    elif game_value == -1:
        result = f"{opponent.title()} wins!"
    else:
        result = "Draw!"
    
    if verbose:
        print(f"Game over: {result}")
    
    return result

def evaluate_model(model_path, num_games=10, opponent='random'):
    """Evaluate a trained model against an opponent."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = TicTacToeNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from iteration {checkpoint['iteration']}")
    
    # Initialize game
    game = TicTacToe()
    
    # Play games
    wins = 0
    losses = 0
    draws = 0
    
    print(f"\nPlaying {num_games} games against {opponent} opponent...")
    
    for game_num in range(num_games):
        print(f"\nGame {game_num + 1}/{num_games}")
        result = play_game(model, game, device, opponent, verbose=False)
        
        if "Model wins" in result:
            wins += 1
        elif "wins" in result:
            losses += 1
        else:
            draws += 1
        
        print(f"Result: {result}")
    
    print(f"\nFinal results: {wins} wins, {losses} losses, {draws} draws")
    if wins + losses + draws > 0:
        print(f"Win rate: {wins/(wins+losses+draws)*100:.1f}%")
    
    return wins, losses, draws

def interactive_game(model_path):
    """Play an interactive game against the model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = TicTacToeNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    game = TicTacToe()
    mcts = MCTS(model, game, num_simulations=100, device=device)
    
    print("Interactive Tic-Tac-Toe Game")
    print("You are X, the model is O")
    print("Enter moves as row,col (e.g., 0,1 for top middle)")
    print()
    
    state = game.get_initial_state()
    current_player = 1  # Human starts
    
    while not game.get_value_and_terminated(state, None)[1]:
        game.display(state)
        
        if current_player == 1:  # Human turn
            while True:
                try:
                    move = input("Your move (row,col): ").strip()
                    row, col = map(int, move.split(','))
                    action = row * 3 + col
                    
                    if 0 <= row < 3 and 0 <= col < 3 and state[row, col] == 0:
                        break
                    else:
                        print("Invalid move! Try again.")
                except:
                    print("Invalid format! Use row,col (e.g., 0,1)")
        else:  # Model turn
            print("Model is thinking...")
            policy = mcts.search(state)
            valid_moves = game.get_valid_moves(state)
            masked_policy = policy * valid_moves
            action = masked_policy.argmax().item()
            row, col = action // 3, action % 3
            print(f"Model plays: {row},{col}")
        
        state = game.get_next_state(state, action, current_player)
        current_player = game.get_opponent(current_player)
    
    game.display(state)
    game_value, _ = game.get_value_and_terminated(state, None)
    
    if game_value == 1:
        print("You win!")
    elif game_value == -1:
        print("Model wins!")
    else:
        print("It's a draw!")

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python evaluate.py <model_path> [num_games] [opponent]")
        print("  python evaluate.py <model_path> interactive")
        print()
        print("Examples:")
        print("  python evaluate.py tictactoe_checkpoint_1.pt 10 random")
        print("  python evaluate.py tictactoe_checkpoint_1.pt 5 optimal")
        print("  python evaluate.py tictactoe_checkpoint_1.pt interactive")
    elif len(sys.argv) == 3 and sys.argv[2] == 'interactive':
        interactive_game(sys.argv[1])
    else:
        model_path = sys.argv[1]
        num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 10
        opponent = sys.argv[3] if len(sys.argv) > 3 else 'random'
        evaluate_model(model_path, num_games, opponent)