import torch
import chess
from model import ChessNet, encode_board
from mcts import MCTS
from move_encoding import index_to_move, get_legal_move_mask

def demo_model():
    """Demonstrate the AlphaZero chess model."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ChessNet().to(device)
    model.eval()
    
    # Initialize MCTS
    mcts = MCTS(model, num_simulations=100, device=device)
    
    # Start a game
    board = chess.Board()
    print("Starting position:")
    print(board)
    print()
    
    move_count = 0
    while not board.is_game_over() and move_count < 10:  # Limit to 10 moves for demo
        print(f"Move {move_count + 1}: {'White' if board.turn == chess.WHITE else 'Black'} to move")
        
        # Get model's move
        policy = mcts.search(board)
        legal_mask = get_legal_move_mask(board)
        masked_policy = policy * legal_mask
        move_idx = masked_policy.argmax().item()
        move = index_to_move(move_idx, board)
        
        if move is None or move not in board.legal_moves:
            print("Model couldn't find a valid move, picking random move")
            import random
            move = random.choice(list(board.legal_moves))
        
        print(f"Model plays: {move}")
        board.push(move)
        print(board)
        print()
        
        move_count += 1
    
    print("Demo completed!")

if __name__ == '__main__':
    demo_model()
