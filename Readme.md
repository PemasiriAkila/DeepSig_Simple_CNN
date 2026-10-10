# RadioML Signal Classification with SimpleCNN

A PyTorch implementation of automatic modulation classification (AMC) using the **RadioML 2018.01A** dataset from [DeepSig](https://www.deepsig.ai/datasets).

## Overview

This project trains a convolutional neural network (`SimpleCNN`) to classify radio signals by their modulation type. Given raw in-phase/quadrature (IQ) samples, the model learns to distinguish between 24 different digital and analog modulation schemes across a range of signal-to-noise ratios (SNR).

## Dataset

**RadioML 2018.01A** — a synthetic dataset of radio signals generated with simulated channel effects.

| Property | Value |
|---|---|
| Modulation classes | 24 |
| Total examples | 2,000,000 |
| Samples per example | 1,024 |
| Format | HDF5 (`.h5`) |

### Download

```bash
wget https://opendata.deepsig.io/datasets/2018.01/2018.01.OSC.0001_1024x2M.h5.tar.gz
tar -xzf 2018.01.OSC.0001_1024x2M.h5.tar.gz
```

The dataset is also mirrored at [github.com/ianblenke/deepsig_dataset](https://github.com/ianblenke/deepsig_dataset).

### Citation

If you use this dataset, please cite:

O’Shea, Timothy James, Tamoghna Roy, and T. Charles Clancy. "Over-the-air deep learning based radio signal classification." IEEE Journal of Selected Topics in Signal Processing 12.1 (2018): 168-179.

## Requirements

```bash
pip install torch torchvision numpy scipy tensorboard
```

The project also depends on the following local modules:

```
networks/
  SimpleCNN.py       # CNN model definition
data/
  Deepsing_Data.py   # Dataset loader (RadioML_2018 class)
utils/
  visualizer.py      # Training logger / scalar printer
```

## Project Structure

```
.
├── demo_simple_gpu.py     # Main training script
├── networks/
│   └── SimpleCNN.py
├── data/
│   └── Deepsing_Data.py
├── utils/
│   └── visualizer.py
└── cachedir/              # Created automatically at runtime
    └── <checkpoint_dir>/
        └── <run_name>/
            ├── *.pth              # Model checkpoints
            └── results/
                └── <epoch>_mod.mat  # Per-epoch test predictions
```

## Configuration

Key hyperparameters are set at the top of `demo_simple_gpu.py`:

| Parameter | Default | Description |
|---|---|---|
| `TRAIN_SPLIT` | 0.7 | Fraction of data used for training |
| `VAL_SPLIT` | 0.2 | Fraction of data used for validation |
| `TEST_SPLIT` | 0.1 | Remaining fraction used for testing |
| `BATCH_SIZE` | 512 | Mini-batch size |
| `n_epochs` | 8 | Number of training epochs |
| `save_epoch_freq` | 1 | Save a checkpoint every N epochs |
| Learning rate | 0.0001 | Adam optimiser learning rate |

## Usage

```bash
python demo_simple_gpu.py
```

Training will automatically use a CUDA GPU if one is available, otherwise falls back to CPU.

### What happens during training

Each epoch:
1. **Train** — forward pass, cross-entropy loss, Adam update, logs loss and modulation accuracy to TensorBoard.
2. **Validate** — evaluates on the held-out validation set, logs loss and accuracy.
3. **Test** — runs inference on the test set and saves ground-truth labels, predictions, and SNR values to a `.mat` file in `cachedir/<checkpoint_dir>/<run_name>/results/`.
4. **Checkpoint** — saves the model weights at every epoch; also saves a separate `min_val_<epoch>.pth` whenever a new best validation loss is achieved.

### Monitoring with TensorBoard

```bash
tensorboard --logdir cachedir/
```

Metrics logged:

- `Loss/train` — per-step training cross-entropy loss  
- `Loss/val` — per-step validation cross-entropy loss  
- `AccuracyMod/train` — per-epoch modulation classification accuracy (training set)  
- `AccuracyMod/val` — per-epoch modulation classification accuracy (validation set)

## Outputs

| Path | Description |
|---|---|
| `cachedir/<run>/min_val_<epoch>.pth` | Best model (lowest validation loss) |
| `cachedir/<run>/<epoch>.pth` | Per-epoch checkpoint |
| `cachedir/<run>/results/<epoch>_mod.mat` | Test set predictions (`.mat`) |

The `.mat` files contain three keys: `gt` (ground-truth modulation labels), `pred` (predicted labels), and `SNR` (per-example SNR values), suitable for plotting per-SNR accuracy curves.

## Model

`SimpleCNN` takes input of shape `(B, 1, 2, 1024)` — a batch of IQ signals reshaped as single-channel 2×1024 images — and outputs class logits over the 24 modulation types.
