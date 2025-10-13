import chess
import torch

def move_to_index(move):
    """Convert chess move to policy index using a more robust encoding."""
    from_square = move.from_square
    to_square = move.to_square
    
    # Calculate direction
    from_rank = from_square // 8
    from_file = from_square % 8
    to_rank = to_square // 8
    to_file = to_square % 8
    
    rank_diff = to_rank - from_rank
    file_diff = to_file - from_file
    
    # Handle knight moves (8 directions)
    knight_moves = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]
    if (rank_diff, file_diff) in knight_moves:
        move_type = 8 + knight_moves.index((rank_diff, file_diff))
        return from_square * 19 + move_type
    
    # Handle queen moves (8 directions)
    queen_moves = [
        (-1, -1), (-1, 0), (-1, 1),
        (0, -1),          (0, 1),
        (1, -1),  (1, 0),  (1, 1)
    ]
    
    # Normalize direction for queen moves
    if rank_diff != 0:
        rank_diff = rank_diff // abs(rank_diff)
    if file_diff != 0:
        file_diff = file_diff // abs(file_diff)
        
    if (rank_diff, file_diff) in queen_moves:
        move_type = queen_moves.index((rank_diff, file_diff))
        return from_square * 19 + move_type
    
    # Handle promotions (other than queen)
    if move.promotion and move.promotion != chess.QUEEN:
        promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        if move.promotion in promo_pieces:
            move_type = 16 + promo_pieces.index(move.promotion)
            return from_square * 19 + move_type
    
    # Handle underpromotions to queen (move_type 0-7)
    if move.promotion == chess.QUEEN:
        # Use the queen move direction
        if rank_diff != 0:
            rank_diff = rank_diff // abs(rank_diff)
        if file_diff != 0:
            file_diff = file_diff // abs(file_diff)
        
        if (rank_diff, file_diff) in queen_moves:
            move_type = queen_moves.index((rank_diff, file_diff))
            return from_square * 19 + move_type
    
    # If we get here, something went wrong
    return 0  # Safe default

def index_to_move(index, board):
    """Convert policy index back to chess move."""
    from_square = index // 19
    move_type = index % 19
    
    # Knight moves
    if 8 <= move_type < 16:
        knight_moves = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        rank_diff, file_diff = knight_moves[move_type - 8]
        from_rank = from_square // 8
        from_file = from_square % 8
        to_rank = from_rank + rank_diff
        to_file = from_file + file_diff
        
        if 0 <= to_rank < 8 and 0 <= to_file < 8:
            to_square = to_rank * 8 + to_file
            move = chess.Move(from_square, to_square)
            if move in board.legal_moves:
                return move
    
    # Queen moves
    elif 0 <= move_type < 8:
        queen_moves = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),          (0, 1),
            (1, -1),  (1, 0),  (1, 1)
        ]
        rank_diff, file_diff = queen_moves[move_type]
        from_rank = from_square // 8
        from_file = from_square % 8
        
        # Try different distances
        for distance in range(1, 8):
            to_rank = from_rank + rank_diff * distance
            to_file = from_file + file_diff * distance
            
            if 0 <= to_rank < 8 and 0 <= to_file < 8:
                to_square = to_rank * 8 + to_file
                move = chess.Move(from_square, to_square)
                if move in board.legal_moves:
                    return move
    
    # Promotions
    elif 16 <= move_type < 19:
        promo_pieces = [chess.KNIGHT, chess.BISHOP, chess.ROOK]
        promotion = promo_pieces[move_type - 16]
        
        # Try all possible promotion moves
        for move in board.legal_moves:
            if (move.from_square == from_square and 
                move.promotion == promotion):
                return move
    
    # If no move found, return None
    return None

def get_legal_move_mask(board):
    """Get a mask of legal moves for the current position."""
    mask = torch.zeros(1968)
    for move in board.legal_moves:
        move_idx = move_to_index(move)
        if 0 <= move_idx < 1968:
            mask[move_idx] = 1
    return mask
