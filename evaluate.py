import torch
import chess
import random
from model import ChessNet, encode_board
from mcts import MCTS
from move_encoding import move_to_index, index_to_move, get_legal_move_mask

def play_game(model, device, verbose=True):
    """Play a game between the model and random moves."""
    board = chess.Board()
    mcts = MCTS(model, num_simulations=100, device=device)
    
    if verbose:
        print("Starting game: Model (White) vs Random (Black)")
        print(board)
        print()
    
    move_count = 0
    while not board.is_game_over() and move_count < 100:
        if board.turn == chess.WHITE:
            # Model plays white
            policy = mcts.search(board)
            legal_mask = get_legal_move_mask(board)
            masked_policy = policy * legal_mask
            move_idx = masked_policy.argmax().item()
            move = index_to_move(move_idx, board)
            if move is None or move not in board.legal_moves:
                move = random.choice(list(board.legal_moves))
            
            if verbose:
                print(f"Model plays: {move}")
        else:
            # Random plays black
            move = random.choice(list(board.legal_moves))
            if verbose:
                print(f"Random plays: {move}")
        
        board.push(move)
        move_count += 1
        
        if verbose:
            print(board)
            print()
    
    # Determine result
    outcome = board.outcome()
    if outcome is None:
        result = "Draw (move limit)"
    elif outcome.winner is None:
        result = "Draw"
    elif outcome.winner == chess.WHITE:
        result = "Model wins!"
    else:
        result = "Random wins!"
    
    if verbose:
        print(f"Game over: {result}")
    
    return result

def evaluate_model(model_path, num_games=5):
    """Evaluate a trained model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    model = ChessNet().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from iteration {checkpoint['iteration']}")
    
    # Play games
    wins = 0
    losses = 0
    draws = 0
    
    for game in range(num_games):
        print(f"\nGame {game + 1}/{num_games}")
        result = play_game(model, device, verbose=False)
        
        if "Model wins" in result:
            wins += 1
        elif "Random wins" in result:
            losses += 1
        else:
            draws += 1
        
        print(f"Result: {result}")
    
    print(f"\nFinal results: {wins} wins, {losses} losses, {draws} draws")
    print(f"Win rate: {wins/num_games*100:.1f}%")

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
        num_games = int(sys.argv[2]) if len(sys.argv) > 2 else 5
        evaluate_model(model_path, num_games)
    else:
        print("Usage: python evaluate.py <model_path> [num_games]")
        print("Example: python evaluate.py model_checkpoint_1.pt 10")
