import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import time
import random

from model import TicTacToeNet
from game import TicTacToe
from mcts import MCTS

class TicTacToeDataset(Dataset):
    def __init__(self, states, policies, values):
        self.states = states
        self.policies = policies
        self.values = values
        
    def __len__(self):
        return len(self.states)
        
    def __getitem__(self, idx):
        return self.states[idx], self.policies[idx], self.values[idx]

def self_play_game(model, game, mcts, device, max_moves=9):
    """Play a single game and return training data."""
    state = game.get_initial_state()
    states, policies, values = [], [], []
    move_count = 0
    current_player = 1
    
    while not game.get_value_and_terminated(state, None)[1] and move_count < max_moves:
        # Get current state
        encoded_state = game.get_encoded_state(state).to(device)
        states.append(encoded_state.cpu())
        
        # Get MCTS policy
        policy = mcts.search(state)
        policies.append(policy.cpu())
        
        # Select move with temperature
        if move_count < 6:  # Temperature = 1 for first 6 moves
            # Apply legal move mask
            valid_moves = game.get_valid_moves(state)
            masked_policy = policy * valid_moves
            masked_policy = masked_policy / masked_policy.sum()
            
            probs = masked_policy.cpu().numpy()
            action = np.random.choice(len(probs), p=probs)
        else:  # Temperature = 0
            # Apply legal move mask and select best move
            valid_moves = game.get_valid_moves(state)
            masked_policy = policy * valid_moves
            action = masked_policy.argmax().item()
        
        # Make the move
        state = game.get_next_state(state, action, current_player)
        current_player = game.get_opponent(current_player)
        move_count += 1
    
    # Get game result
    game_value, _ = game.get_value_and_terminated(state, None)
    
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

def evaluate_model(model, game, device, num_games=10):
    """Evaluate the model by playing against random moves."""
    model.eval()
    wins = 0
    losses = 0
    draws = 0
    
    for game_num in range(num_games):
        state = game.get_initial_state()
        mcts = MCTS(model, game, num_simulations=50, device=device)
        current_player = 1
        
        while not game.get_value_and_terminated(state, None)[1]:
            if current_player == 1:
                # Model plays
                policy = mcts.search(state)
                valid_moves = game.get_valid_moves(state)
                masked_policy = policy * valid_moves
                action = masked_policy.argmax().item()
            else:
                # Random plays
                valid_moves = game.get_valid_moves(state)
                valid_actions = np.where(valid_moves == 1)[0]
                if len(valid_actions) > 0:
                    action = np.random.choice(valid_actions)
                else:
                    break  # No valid moves, game should be over
            
            state = game.get_next_state(state, action, current_player)
            current_player = game.get_opponent(current_player)
        
        # Evaluate result
        game_value, _ = game.get_value_and_terminated(state, None)
        if game_value == 1:  # Model won
            wins += 1
        elif game_value == -1:  # Random won
            losses += 1
        else:  # Draw
            draws += 1
    
    return wins, losses, draws

def train_model(num_iterations=5, games_per_iteration=100, num_epochs=5, batch_size=64, num_simulations=50):
    """Main training loop."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize game and model
    game = TicTacToe()
    model = TicTacToeNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    mcts = MCTS(model, game, num_simulations=num_simulations, device=device)
    
    for iteration in range(num_iterations):
        print(f"\nIteration {iteration + 1}/{num_iterations}")
        
        # Self-play phase
        all_states = []
        all_policies = []
        all_values = []
        
        print("Generating self-play games...")
        start_time = time.time()
        
        for game_num in tqdm(range(games_per_iteration)):
            states, policies, values = self_play_game(model, game, mcts, device)
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
        dataset = TicTacToeDataset(states_tensor, policies_tensor, values_tensor)
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
        wins, losses, draws = evaluate_model(model, game, device)
        print(f"Evaluation: {wins} wins, {losses} losses, {draws} draws")
        print(f"Win rate: {wins/(wins+losses+draws)*100:.1f}%")
        
        # Save model checkpoint
        torch.save({
            'iteration': iteration + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'wins': wins,
            'losses': losses,
            'draws': draws,
        }, f'tictactoe_checkpoint_{iteration + 1}.pt')
        
        print(f"Saved model checkpoint {iteration + 1}")

if __name__ == '__main__':
    train_model()