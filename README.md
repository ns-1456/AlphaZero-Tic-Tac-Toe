# AlphaZero-like Chess Model

A PyTorch implementation of an AlphaZero-like model for playing chess, featuring:

- Deep neural network with policy and value heads
- Monte Carlo Tree Search (MCTS) with PUCT algorithm
- Self-play training pipeline
- Improved move encoding and board representation
- Model evaluation and testing tools

## Requirements

```
numpy>=1.21.0
torch>=1.9.0
python-chess>=1.0.0
tqdm>=4.62.0
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

## Project Structure

- `model.py`: Neural network architecture (ResNet with policy and value heads)
- `mcts.py`: Monte Carlo Tree Search implementation with PUCT
- `move_encoding.py`: Robust move encoding/decoding functions
- `improved_train.py`: Enhanced training script with better self-play
- `evaluate.py`: Model evaluation against random play
- `demo.py`: Simple demonstration of the model
- `train.py`: Original training script
- `simple_training.py`: Simplified training for quick testing

## Quick Start

### 1. Run a Demo
```bash
python demo.py
```

### 2. Train a Model
```bash
python improved_train.py
```

### 3. Evaluate a Trained Model
```bash
python evaluate.py model_checkpoint_1.pt 10
```

## Model Architecture

- **Input**: 8x8x19 board representation
  - 12 piece planes (6 piece types × 2 colors)
  - 4 castling rights planes
  - 1 side to move plane
  - 1 move count plane
  - 1 no-progress count plane
- **Backbone**: ResNet with 19 residual blocks
- **Policy head**: Predicts move probabilities (1968 possible moves)
- **Value head**: Evaluates position (-1 to 1)

## Training Process

1. **Self-play**: Generate games using MCTS and current model
2. **Data collection**: Collect (state, policy, value) training data
3. **Training**: Train model on collected data
4. **Evaluation**: Test model performance
5. **Repeat**: Continue with improved model

## Key Improvements

- **Robust move encoding**: Handles all chess moves including promotions and special moves
- **Legal move masking**: Ensures model only considers legal moves
- **Better board representation**: Includes castling rights, move count, and other game state
- **Improved MCTS**: Better handling of game outcomes and value propagation
- **Evaluation tools**: Test model performance against random play

## Training on Google Colab

1. Upload files to your Google Drive
2. Create a new Colab notebook
3. Mount your Google Drive:
```python
from google.colab import drive
drive.mount('/content/drive')
```

4. Install dependencies:
```python
!pip install -r requirements.txt
```

5. Start training:
```python
!python improved_train.py
```

## Checkpoints

Model checkpoints are saved after each iteration in format: `model_checkpoint_N.pt`

Each checkpoint includes:
- Model state dictionary
- Optimizer state dictionary
- Training iteration number
- Evaluation results (wins, losses, draws)

## Usage Examples

### Basic Training
```python
from improved_train import train_model
train_model(num_iterations=5, games_per_iteration=20)
```

### Model Evaluation
```python
from evaluate import evaluate_model
evaluate_model('model_checkpoint_1.pt', num_games=10)
```

### Playing a Game
```python
from demo import demo_model
demo_model()
```

## Performance Notes

- Training is computationally intensive and benefits from GPU acceleration
- Self-play games can take several minutes each depending on MCTS simulations
- Model performance improves with more training iterations and games per iteration
- Recommended: Start with small parameters for testing, then scale up

## Contributing

Feel free to submit issues and enhancement requests!