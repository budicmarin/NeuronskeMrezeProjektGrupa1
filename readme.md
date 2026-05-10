# Time Series Classification: CNN1D vs GRU

## Project Overview

This project implements and compares two deep learning architectures for time series classification on automotive sensor data. We train and evaluate **CNN1D** (1D Convolutional Neural Network) and **GRU** (Gated Recurrent Unit) models on the FordA and FordB datasets.

## Objective

Develop and benchmark two distinct neural network models for classifying automotive engine condition time series data, evaluating their performance across different test distributions.

## Models

- **CNN1D**: 1D convolutional architecture with batch normalization, max pooling, and adaptive average pooling for feature extraction
- **GRU**: Recurrent architecture using Gated Recurrent Units to capture temporal dependencies in sequences

## Datasets

- **FordA_TRAIN**: 3,601 samples (1,846 negative, 1,755 positive) - Training data
- **FordA_TEST**: 1,320 samples (681 negative, 639 positive) - In-distribution test set
- **FordB_TEST**: 810 samples (401 negative, 409 positive) - Out-of-distribution test set
- Each sequence: 500 time steps

Labels: -1 (class 0) and 1 (class 1), converted internally to 0 and 1.

## Results Summary

### CNN1D Performance
- **FordA_TEST**: Accuracy 0.925, Precision 0.901, Recall 0.976, F1 0.937
- **FordB_TEST**: Accuracy 0.801, Precision 0.741, Recall 0.895, F1 0.813

### GRU Performance
- **FordA_TEST**: Accuracy 0.537, Precision 0.498, Recall 0.371, F1 0.425
- **FordB_TEST**: Accuracy 0.519, Precision 0.463, Recall 0.330, F1 0.386

**Conclusion**: CNN1D significantly outperforms GRU on both test sets, with particularly strong generalization to the out-of-distribution FordB dataset.

## Project Structure

```
.
├── data.py                 # Data loading and preprocessing
├── models.py              # CNN1D and GRU model definitions
├── train.py               # Training pipeline, validation, and testing
├── plot.py                # Visualization and plotting utilities
├── ckpt/                  # Model checkpoints and configurations
│   ├── CNN1D.pt
│   ├── CNN1D.json
│   ├── GRU.pt
│   └── GRU.json
├── results/               # Aggregated results
│   ├── metrics.json
│   └── histories.json
├── plots/                 # Generated visualizations
│   ├── CNN1D_hist.png
│   ├── CNN1D_FordA_TEST_cm.png
│   ├── CNN1D_FordB_TEST_cm.png
│   ├── GRU_hist.png
│   ├── GRU_FordA_TEST_cm.png
│   ├── GRU_FordB_TEST_cm.png
│   └── comparison.png
├── FordA/                 # FordA dataset
└── FordB/                 # FordB dataset
```

## Training Configuration

- **Epochs**: 50 (with early stopping, patience=10)
- **Learning Rate**: 1e-3 (Adam optimizer with ReduceLROnPlateau scheduling)
- **Batch Size**: 64
- **Validation Split**: 20%
- **Gradient Clipping**: Applied to GRU (clip_value=1.0)
- **Dropout**: 0.5 (feature extraction), 0.3 (classifier)

## Usage

```bash
python train.py
```

This runs the complete pipeline:
1. Loads and normalizes data (z-score normalization on training set)
2. Trains both models with early stopping
3. Evaluates on both test sets
4. Saves checkpoints and metrics to `ckpt/` and `results/`
5. Generates visualizations in `plots/`

## Dependencies

- PyTorch
- NumPy
- Scikit-learn
- Matplotlib

## Key Findings

- CNN1D is superior for this time series classification task, achieving 92.5% accuracy on FordA_TEST
- CNN1D generalizes well to the out-of-distribution FordB_TEST (80.1% accuracy)
- GRU struggles with sequence modeling in this domain, suggesting temporal dependencies are less critical than local feature patterns
- Batch normalization and max pooling in CNN1D effectively extract discriminative features
- The significant accuracy drop on FordB_TEST for both models indicates domain shift between datasets