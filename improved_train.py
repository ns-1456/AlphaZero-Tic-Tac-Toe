import torch
import chess
import numpy as np
from tqdm import tqdm
from model import ChessNet, encode_board
from mcts import MCTS
from move_encoding import move_to_index, index_to_move, get_legal_move_mask
import random
from torch.utils.data import Dataset, DataLoader
import time

class ChessDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def self_play_game(model, mcts, device, max_moves=200):
    """Play a single game and return training data."""
    board = chess.Board()
    states, policies, values = [], [], []
    move_count = 0
    
    while not board.is_game_over() and move_count < max_moves:
        # Get current state
        state_tensor = encode_board(board).to(device)
        states.append(state_tensor.cpu())
        
        # Get MCTS policy
        policy = mcts.search(board)
        policies.append(policy.cpu())
        
        # Select move with temperature
        if len(board.move_stack) < 30:  # Temperature = 1
            # Apply legal move mask
            legal_mask = get_legal_move_mask(board)
            masked_policy = policy * legal_mask
            masked_policy = masked_policy / masked_policy.sum()
            
            probs = masked_policy.cpu().numpy()
            move_idx = np.random.choice(len(probs), p=probs)
        else:  # Temperature = 0
            # Apply legal move mask and select best move
            legal_mask = get_legal_move_mask(board)
            masked_policy = policy * legal_mask
            move_idx = masked_policy.argmax().item()
        
        # Convert move index back to chess move
        move = index_to_move(move_idx, board)
        if move is None or move not in board.legal_moves:
            # Fallback: pick a random legal move
            move = random.choice(list(board.legal_moves))
        
        board.push(move)
        move_count += 1
    
    # Get game result
    outcome = board.outcome()
    if outcome is None:
        game_value = 0  # Draw by move limit
    elif outcome.winner is None:
        game_value = 0  # Draw
    else:
        # Winner from the perspective of the starting player (white)
        game_value = 1 if outcome.winner == chess.WHITE else -1
    
    # Assign values to all states (from the perspective of the player to move)
    current_value = game_value
    for i in range(len(states)):
        # Value flips for each player
        values.append(current_value)
        current_value = -current_value
        
    return states, policies, values

def train_epoch(model, optimizer, dataloader, device):
    """Train model for one epoch."""
    model.train()
    total_loss = 0
    policy_loss_total = 0
    value_loss_total = 0
    num_batches = 0
    
    for states, policies, values in dataloader:
        states = states.to(device)
        policies = policies.to(device)
        values = values.to(device)
        
        # Forward pass
        policy_pred, value_pred = model(states)
        
        # Calculate loss
        policy_loss = -torch.sum(policies * torch.log_softmax(policy_pred, dim=1)) / policies.shape[0]
        value_loss = torch.mean((value_pred.squeeze() - values) ** 2)
        loss = policy_loss + value_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        policy_loss_total += policy_loss.item()
        value_loss_total += value_loss.item()
        num_batches += 1
    
    return total_loss / num_batches, policy_loss_total / num_batches, value_loss_total / num_batches

def evaluate_model(model, device, num_games=10):
    """Evaluate the model by playing against random moves."""
    model.eval()
    wins = 0
    losses = 0
    draws = 0
    
    for game in range(num_games):
        board = chess.Board()
        mcts = MCTS(model, num_simulations=100, device=device)
        
        while not board.is_game_over():
            if board.turn == chess.WHITE:
                # Model plays white
                policy = mcts.search(board)
                legal_mask = get_legal_move_mask(board)
                masked_policy = policy * legal_mask
                move_idx = masked_policy.argmax().item()
                move = index_to_move(move_idx, board)
                if move is None or move not in board.legal_moves:
                    move = random.choice(list(board.legal_moves))
            else:
                # Random plays black
                move = random.choice(list(board.legal_moves))
            
            board.push(move)
        
        # Evaluate result
        outcome = board.outcome()
        if outcome is None:
            draws += 1
        elif outcome.winner is None:
            draws += 1
        elif outcome.winner == chess.WHITE:
            wins += 1
        else:
            losses += 1
    
    return wins, losses, draws

def train_model(num_iterations=10, games_per_iteration=50, num_epochs=5, batch_size=32, num_simulations=200):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize model
    model = ChessNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    mcts = MCTS(model, num_simulations=num_simulations, device=device)
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Self-play phase
        all_states = []
        all_policies = []
        all_values = []
        
        print("Generating self-play games...")
        start_time = time.time()
        
        for game in tqdm(range(games_per_iteration)):
            states, policies, values = self_play_game(model, mcts, device)
            all_states.extend(states)
            all_policies.extend(policies)
            all_values.extend(values)
        
        elapsed = time.time() - start_time
        print(f"Self-play completed in {elapsed:.1f} seconds. Collected {len(all_states)} positions.")
        
        # Convert to tensors
        states_tensor = torch.stack(all_states)
        policies_tensor = torch.stack(all_policies)
        values_tensor = torch.tensor(all_values, dtype=torch.float32)
        
        # Create dataset and dataloader
        dataset = ChessDataset(states_tensor, policies_tensor, values_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training phase
        print("Training on collected data...")
        for epoch in range(num_epochs):
            total_loss, policy_loss, value_loss = train_epoch(model, optimizer, dataloader, device)
            print(f"Epoch {epoch + 1}/{num_epochs} - "
                  f"Loss: {total_loss:.4f} "
                  f"(Policy: {policy_loss:.4f}, Value: {value_loss:.4f})")
        
        # Evaluate model
        print("Evaluating model...")
        wins, losses, draws = evaluate_model(model, device)
        print(f"Evaluation: {wins} wins, {losses} losses, {draws} draws")
        
        # Save model checkpoint
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wins': wins,
            'losses': losses,
            'draws': draws,
        }, f'model_checkpoint_{iteration + 1}.pt')
        
        print(f"Saved model checkpoint {iteration + 1}")

if __name__ == '__main__':
    train_model()
