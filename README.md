# AlphaZero Tic-Tac-Toe Implementation

A PyTorch implementation of AlphaZero for Tic-Tac-Toe, featuring:

- Deep neural network with policy and value heads
- Monte Carlo Tree Search (MCTS) with PUCT algorithm
- Self-play training pipeline
- Clean game abstraction for easy extension
- Interactive tutorials and demos

## Why Tic-Tac-Toe?

While AlphaZero is famous for mastering Chess, Go, and Shogi, Tic-Tac-Toe provides an excellent learning environment:

- **Fast Training**: Minutes instead of hours/days
- **Verifiable Learning**: Can check against optimal play
- **Simple State Space**: Easy to understand and visualize
- **Educational Value**: Perfect for learning AlphaZero concepts

## Requirements

```
numpy>=1.21.0
torch>=1.9.0
tqdm>=4.62.0
matplotlib>=3.0.0  # For visualizations
```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ns-1456/AlphaZero-Chess.git
cd AlphaZero-Chess
```

2. Create a virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### 1. Run a Demo
```bash
python demo.py
```

### 2. Train a Model
```bash
python train.py
```

### 3. Evaluate a Trained Model
```bash
python evaluate.py tictactoe_checkpoint_1.pt 10 random
```

### 4. Play Interactive Game
```bash
python evaluate.py tictactoe_checkpoint_1.pt interactive
```

## Project Structure

- `game.py`: Game abstraction layer with TicTacToe implementation
- `model.py`: Neural network architecture (TicTacToeNet)
- `mcts.py`: Monte Carlo Tree Search implementation
- `train.py`: Training script with self-play pipeline
- `evaluate.py`: Model evaluation against random/optimal play
- `demo.py`: Interactive demonstrations
- `tutorial.ipynb`: Step-by-step tutorial (Jupyter notebook)
- `training.ipynb`: Interactive training interface

## Model Architecture

- **Input**: 3x3x3 board representation
  - Plane 0: Current player's pieces
  - Plane 1: Opponent's pieces  
  - Plane 2: Turn indicator
- **Backbone**: ResNet with 4 residual blocks
- **Policy head**: Predicts move probabilities (9 possible moves)
- **Value head**: Evaluates position (-1 to 1)

## Training Process

1. **Self-play**: Generate games using MCTS and current model
2. **Data collection**: Collect (state, policy, value) training data
3. **Training**: Train model on collected data
4. **Evaluation**: Test model performance
5. **Repeat**: Continue with improved model

## Expected Performance

- **Training time**: 5-10 minutes on CPU
- **Win rate vs random**: 90%+ after 2-3 iterations
- **Win rate vs optimal**: 60-80% (Tic-Tac-Toe is mostly draws with optimal play)

## Usage Examples

### Basic Training
```python
from train import train_model
train_model(num_iterations=5, games_per_iteration=100)
```

### Model Evaluation
```python
from evaluate import evaluate_model
wins, losses, draws = evaluate_model('tictactoe_checkpoint_1.pt', num_games=20)
```

### Interactive Play
```python
from evaluate import interactive_game
interactive_game('tictactoe_checkpoint_1.pt')
```

## Key Features

### Game Abstraction
The `Game` class provides a clean interface that can be extended to other games:
```python
class Game(ABC):
    def get_initial_state(self): pass
    def get_next_state(self, state, action, player): pass
    def get_valid_moves(self, state): pass
    def get_value_and_terminated(self, state, action): pass
    def get_encoded_state(self, state): pass
```

### MCTS with PUCT
Monte Carlo Tree Search uses the PUCT algorithm for balanced exploration/exploitation:
- **Selection**: Choose moves using UCB scores
- **Expansion**: Add new nodes for unexplored moves
- **Simulation**: Use neural network for evaluation
- **Backpropagation**: Update values up the tree

### Self-Play Training
The model improves through self-play:
1. Current model plays against itself using MCTS
2. Game outcomes provide training targets
3. Neural network learns from collected data
4. Process repeats with improved model

## Challenges & Solutions

### Challenge 1: Exploration vs Exploitation
**Problem**: Model might get stuck in local optima
**Solution**: Use temperature decay and exploration noise

### Challenge 2: Value Propagation
**Problem**: Incorrect value assignment to intermediate positions
**Solution**: Proper backpropagation in MCTS with alternating perspectives

### Challenge 3: Training Stability
**Problem**: Loss curves might be unstable
**Solution**: Use lower learning rate (0.001) and proper normalization

### Challenge 4: Overfitting
**Problem**: Model memorizes instead of generalizing
**Solution**: Add data augmentation (board rotations/reflections)

## Extending to Other Games

The framework is designed to be easily extensible:

1. **Create new game class**: Extend `Game` abstract class
2. **Implement game logic**: Define state representation and rules
3. **Adjust neural network**: Modify input/output dimensions
4. **Tune hyperparameters**: Adjust MCTS simulations and training params

Example games that could be added:
- Connect Four
- Othello/Reversi
- Checkers
- Simple card games

## Performance Tips

1. **Start small**: Use fewer simulations and games initially
2. **Monitor progress**: Check win rates against random play
3. **Adjust parameters**: Tune learning rate and MCTS simulations
4. **Use GPU**: Training is much faster with CUDA support

## Troubleshooting

### Model doesn't improve
- Check if MCTS is working correctly
- Verify value propagation in backpropagation
- Try different learning rates

### Training is too slow
- Reduce number of MCTS simulations
- Use smaller batch sizes
- Enable GPU acceleration

### Poor performance against optimal play
- Tic-Tac-Toe with optimal play is mostly draws
- Focus on win rate against random play
- Consider Connect Four for more interesting results

## Contributing

Feel free to submit issues and enhancement requests! Some ideas:
- Add Connect Four implementation
- Implement data augmentation
- Add visualization tools
- Create more sophisticated evaluation metrics

## References

This implementation is inspired by:
- [AlphaZero paper](https://arxiv.org/abs/1712.01815)
- [foersterrobert's AlphaZero](https://github.com/foersterrobert/AlphaZero)
- [AlphaZero from scratch tutorial](https://github.com/foersterrobert/AlphaZeroFromScratch)

## License

This project is open source and available under the MIT License.