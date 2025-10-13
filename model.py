import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class ResBlock(nn.Module):
    def __init__(self, channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += identity
        x = F.relu(x)
        return x

class TicTacToeNet(nn.Module):
    def __init__(self):
        super(TicTacToeNet, self).__init__()
        
        # Input: 3x3x3 (much simpler than chess)
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual blocks (fewer than chess)
        self.res_blocks = nn.ModuleList([ResBlock(64) for _ in range(4)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(64, 32, 3, padding=1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * 3 * 3, 9)  # 9 possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(64, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(3 * 3, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Initial convolution block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 32 * 3 * 3)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 3 * 3)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

# Keep the old ChessNet for backward compatibility
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        
        # Input: 8x8x19 (12 piece types + 7 auxiliary channels)
        self.conv1 = nn.Conv2d(19, 256, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        
        # Residual blocks
        self.res_blocks = nn.ModuleList([ResBlock(256) for _ in range(19)])
        
        # Policy head
        self.policy_conv = nn.Conv2d(256, 256, 3, padding=1)
        self.policy_bn = nn.BatchNorm2d(256)
        self.policy_fc = nn.Linear(256 * 8 * 8, 1968)  # All possible moves
        
        # Value head
        self.value_conv = nn.Conv2d(256, 1, 1)
        self.value_bn = nn.BatchNorm2d(1)
        self.value_fc1 = nn.Linear(8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

    def forward(self, x):
        # Initial convolution block
        x = F.relu(self.bn1(self.conv1(x)))
        
        # Residual tower
        for block in self.res_blocks:
            x = block(x)
        
        # Policy head
        policy = F.relu(self.policy_bn(self.policy_conv(x)))
        policy = policy.view(-1, 256 * 8 * 8)
        policy = self.policy_fc(policy)
        
        # Value head
        value = F.relu(self.value_bn(self.value_conv(x)))
        value = value.view(-1, 8 * 8)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value

def encode_board(board):
    """Convert chess board to input tensor with 19 channels."""
    import chess
    piece_chars = 'pnbrqkPNBRQK'
    piece_map = {piece: i for i, piece in enumerate(piece_chars)}
    
    # Initialize 19 planes of 8x8
    planes = torch.zeros(19, 8, 8)
    
    # Fill in piece positions (12 channels)
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            rank = square // 8
            file = square % 8
            piece_idx = piece_map[piece.symbol()]
            planes[piece_idx][rank][file] = 1
    
    # Castling rights (4 channels)
    planes[12][0][0] = float(board.has_kingside_castling_rights(chess.WHITE))
    planes[13][0][0] = float(board.has_queenside_castling_rights(chess.WHITE))
    planes[14][0][0] = float(board.has_kingside_castling_rights(chess.BLACK))
    planes[15][0][0] = float(board.has_queenside_castling_rights(chess.BLACK))
    
    # Side to move (1 channel)
    if board.turn == chess.WHITE:
        planes[16].fill_(1)
    
    # Move count (1 channel)
    planes[17].fill_(min(board.fullmove_number / 50.0, 1.0))
    
    # No-progress count (1 channel)
    planes[18].fill_(min(board.halfmove_clock / 100.0, 1.0))
    
    return planes