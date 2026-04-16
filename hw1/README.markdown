# HW1: 三层 MLP 分类器 — EuroSAT 地表覆盖图像分类

## 环境依赖

```
Python >= 3.10
NumPy
Pillow
Matplotlib
```

安装：

```bash
pip install numpy pillow matplotlib
```

## 运行步骤

### 1. 超参数搜索（可选，找最优配置）

```bash
# 随机搜索 20 组，每组训练 30 epoch
python main.py search --search_type random --n_trials 20 --search_epochs 30

# 或网格搜索（组合较多，耗时更长）
python main.py search --search_type grid --search_epochs 30
```

搜索结果保存在 `outputs/search_results.json`，最优模型自动保存为 `outputs/best_model.npz`。

### 2. 训练（使用默认参数或搜索得到的最优参数）

```bash
# 默认参数
python main.py train --hidden_dim 512 --activation relu --lr 0.01 --reg 1e-4 --epochs 50

# 根据搜索结果调整参数，例如：
# python main.py train --hidden_dim 1024 --activation relu --lr 0.03 --reg 5e-4 --epochs 80
```

训练过程会自动保存最优模型（`outputs/best_model.npz`）和训练曲线图（`outputs/training_curves.png`）。

### 3. 测试评估

```bash
python main.py test
```

输出测试集准确率、各类别准确率、混淆矩阵，并生成：
- `outputs/confusion_matrix.png` — 混淆矩阵热力图
- `outputs/error_analysis.png` — 分类错误样本可视化

### 4. 权重可视化

```bash
python main.py visualize
```

生成：
- `outputs/training_curves.png` — train/val loss 曲线 + val accuracy 曲线
- `outputs/weight_visualization.png` — 第一层隐藏层权重 reshape 成 64x64x3 图像

## 全部参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hidden_dim` | 512 | 隐藏层维度 |
| `--activation` | relu | 激活函数 (relu / sigmoid / tanh) |
| `--lr` | 0.01 | 初始学习率 |
| `--lr_decay` | 0.95 | 学习率衰减系数 |
| `--lr_step` | 10 | 每多少 epoch 衰减一次 |
| `--reg` | 1e-4 | L2 正则化强度 |
| `--batch_size` | 128 | Mini-batch 大小 |
| `--epochs` | 50 | 训练轮数 |
| `--seed` | 42 | 随机种子 |

## 输出文件

所有输出保存在 `outputs/` 目录下：

```
outputs/
├── best_model.npz            # 最优模型权重
├── history.npz               # 训练历史数据
├── norm_params.npz           # 标准化参数 (mean, std)
├── search_results.json       # 超参数搜索结果
├── training_curves.png       # 训练曲线图
├── confusion_matrix.png      # 混淆矩阵热力图
├── weight_visualization.png  # 权重可视化
└── error_analysis.png        # 错例分析图
```