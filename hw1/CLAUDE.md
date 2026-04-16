# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

University deep learning homework (HW1): Build a **3-layer MLP classifier from scratch** (no PyTorch/TensorFlow/JAX) for land-cover satellite image classification on the **EuroSAT RGB** dataset. Only NumPy is allowed for matrix operations.

## Dataset

- **EuroSAT_RGB/**: 10 classes of satellite images (64×64 RGB JPEGs), ~3000 images per class
- Classes: AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, SeaLake
- Must split into train/validation/test sets

## Assignment Constraints (Critical)

- **No autograd frameworks** — must implement forward pass, backward pass (backpropagation), and gradient computation manually using NumPy
- Must support configurable hidden dimension, at least two activation functions (ReLU + Sigmoid/Tanh)
- Required optimizer: SGD with learning rate decay, cross-entropy loss, L2 regularization (weight decay)
- Must save best model weights based on validation accuracy
- Must implement hyperparameter search (grid or random) over learning rate, hidden size, regularization strength

## Required Code Modules

The submission must contain at least these five modules:
1. **Data loading & preprocessing** — load images, normalize, train/val/test split
2. **Model definition** — 3-layer MLP with manual forward/backward
3. **Training loop** — SGD, LR decay, cross-entropy, L2 reg, best-model checkpointing
4. **Test evaluation** — load best weights, report accuracy, print confusion matrix
5. **Hyperparameter search** — grid/random search with logged results

## Commands

```bash
# 训练 (默认参数)
python main.py train --hidden_dim 512 --lr 0.01 --epochs 50 --activation relu

# 超参数搜索 (随机搜索 20 组 / 网格搜索)
python main.py search --search_type random --n_trials 20 --search_epochs 30
python main.py search --search_type grid --search_epochs 30

# 测试评估 (加载最优模型，输出准确率+混淆矩阵+错例图)
python main.py test

# 生成可视化 (训练曲线 + 权重可视化)
python main.py visualize
```

All outputs saved to `outputs/` directory.

## Architecture

- **data_loader.py** — `load_data()`, `normalize()`, `stratified_split()`, `batch_iterator()`
- **model.py** — `MLP` class (Input→Hidden→Output, manual forward/backward), activation classes (`ReLU`/`Sigmoid`/`Tanh`), `softmax()`
- **train.py** — `Trainer` class: SGD, cross-entropy loss, L2 reg, step LR decay, best-model checkpointing
- **test.py** — `evaluate()`, `confusion_matrix()`, `find_misclassified()`
- **search.py** — `grid_search()`, `random_search()`
- **visualize.py** — training curves, weight visualization, confusion matrix heatmap, error analysis plot
- **main.py** — CLI entry point with `train`/`test`/`search`/`visualize` subcommands

## Dependencies

Python 3.10+, NumPy, Pillow, Matplotlib (no deep learning frameworks)

## Deliverables

- Training/validation loss curves and validation accuracy curve (matplotlib)
- First hidden layer weight visualization (reshaped to image dimensions)
- Error analysis of misclassified test samples
- Code on public GitHub with README (environment, how to run train/test)
- Model weights uploaded to Google Drive (linked in report)