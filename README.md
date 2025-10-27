# LeNet-5 Neural Network

This project implements the classic LeNet-5 Convolutional Neural Network architecture for digit recognition on the MNIST dataset.

## Overview

LeNet-5 is a pioneering convolutional neural network designed by Yann LeCun et al. in 1998. This implementation uses PyTorch and includes:

- **LeNet-5 architecture**: 2 convolutional layers + 3 fully connected layers
- **Automatic data download**: MNIST dataset (60,000 training + 10,000 test images)
- **Training pipeline**: Complete training loop with progress tracking
- **Visualization**: Training history plots and prediction examples
- **Model saving**: Trained model saved for later use

## Architecture

```
Input (1x28x28)
    ↓
Conv1 (6 filters, 5x5) + ReLU + MaxPool(2x2)
    ↓
Conv2 (16 filters, 5x5) + ReLU + MaxPool(2x2)
    ↓
Flatten
    ↓
FC1 (120 neurons) + ReLU
    ↓
FC2 (84 neurons) + ReLU
    ↓
FC3 (10 neurons) - Output
```

## Installation

1. Create a virtual environment (recommended):
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Simply run the script:
```bash
python lenet.py
```

The script will automatically:
1. Download the MNIST dataset to `./data/` directory
2. Train the LeNet-5 model for 10 epochs
3. Evaluate on the test set
4. Save the trained model as `lenet5_mnist.pth`
5. Generate visualizations:
   - `training_history.png` - Training/test loss and accuracy curves
   - `predictions.png` - Sample predictions on test images

## Expected Performance

- **Training time**: ~2-5 minutes on CPU, <1 minute on GPU
- **Test accuracy**: ~98-99%
- **Parameters**: ~61,000 trainable parameters

## Customization

You can modify hyperparameters in the `main()` function:

```python
batch_size = 64        # Batch size for training
learning_rate = 0.001  # Learning rate for optimizer
epochs = 10            # Number of training epochs
```

## Output Files

- `lenet5_mnist.pth` - Trained model weights
- `training_history.png` - Training metrics visualization
- `predictions.png` - Sample predictions visualization
- `data/` - MNIST dataset directory (created automatically)

## Requirements

- Python 3.7+
- PyTorch 2.0+
- torchvision
- matplotlib
- numpy

## License

See LICENSE file for details.

