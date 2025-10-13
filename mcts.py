import math
import torch
import numpy as np
from game import Game

class Node:
    def __init__(self, prior=0.0):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.state = None
        self.to_play = 1  # Current player (1 or -1)

    def value(self):
        if self.visit_count == 0:
            return 0
        return self.value_sum / self.visit_count

    def expand(self, state, to_play, policy, game):
        """Expand node with new state and policy."""
        self.state = state
        self.to_play = to_play
        
        # Get legal move mask
        valid_moves = game.get_valid_moves(state)
        
        # Mask the policy to only include legal moves
        masked_policy = policy * valid_moves
        
        # Normalize the masked policy
        policy_sum = masked_policy.sum()
        if policy_sum > 0:
            masked_policy = masked_policy / policy_sum
        else:
            # If all probabilities are zero, use uniform distribution over legal moves
            masked_policy = valid_moves / valid_moves.sum()
        
        # Add children for each legal move
        for action in range(game.action_size):
            if valid_moves[action] == 1:  # Valid move
                prob = masked_policy[action].item()
                self.children[action] = Node(prior=prob)

    def select_child(self):
        """Select child node using PUCT algorithm."""
        c_puct = 2.0
        
        best_score = float('-inf')
        best_action = None
        best_child = None

        # Calculate UCB score for each child
        for action, child in self.children.items():
            ucb_score = child.get_ucb_score(self.visit_count, c_puct)
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child

        return best_action, best_child

    def get_ucb_score(self, parent_visit_count, c_puct):
        """Calculate UCB score for node selection."""
        prior_score = c_puct * self.prior * math.sqrt(parent_visit_count) / (1 + self.visit_count)
        value_score = -self.value() if self.visit_count > 0 else 0
        return prior_score + value_score

class MCTS:
    def __init__(self, model, game, num_simulations=50, device='cpu'):
        self.model = model
        self.game = game
        self.num_simulations = num_simulations
        self.device = device

    def search(self, state):
        """Perform MCTS search and return policy."""
        root = Node(0)
        
        # Evaluate root state
        encoded_state = self.game.get_encoded_state(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            policy, value = self.model(encoded_state)
            policy = torch.softmax(policy, dim=1).squeeze()
        
        # Expand root node
        root.expand(state, 1, policy, self.game)  # Assume current player is 1
        
        # Run simulations
        for _ in range(self.num_simulations):
            node = root
            search_path = [node]
            current_state = state.copy()
            current_player = 1
            
            # Selection
            while node.children and not self.game.get_value_and_terminated(current_state, None)[1]:
                action, node = node.select_child()
                current_state = self.game.get_next_state(current_state, action, current_player)
                current_player = self.game.get_opponent(current_player)
                search_path.append(node)
            
            # Expansion and evaluation
            if not self.game.get_value_and_terminated(current_state, None)[1]:
                encoded_state = self.game.get_encoded_state(current_state).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    policy, value = self.model(encoded_state)
                    policy = torch.softmax(policy, dim=1).squeeze()
                node.expand(current_state, current_player, policy, self.game)
                value = value.item()
            else:
                # Game is over, use actual outcome
                game_value, _ = self.game.get_value_and_terminated(current_state, None)
                value = game_value
            
            # Backpropagation
            for node in reversed(search_path):
                node.value_sum += value if node.to_play == 1 else -value
                node.visit_count += 1
                value = -value  # Value flips between layers
        
        # Calculate policy from visit counts
        policy = torch.zeros(self.game.action_size)
        for action, child in root.children.items():
            policy[action] = child.visit_count
        
        # Normalize policy
        if policy.sum() > 0:
            policy = policy / policy.sum()
        else:
            # If no visits, use uniform distribution over valid moves
            valid_moves = self.game.get_valid_moves(state)
            policy = valid_moves / valid_moves.sum()
            
        return policy