import numpy as np
import os
from PIL import Image


CLASS_NAMES = [
    'AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway',
    'Industrial', 'Pasture', 'PermanentCrop', 'Residential',
    'River', 'SeaLake'
]


def load_data(data_dir='EuroSAT_RGB'):
    """加载全部图像，返回展平的像素矩阵和标签。

    Returns:
        X: (N, 12288) float32, 像素值 [0, 1]
        y: (N,) int, 类别索引 0-9
    """
    images, labels = [], []
    for label_idx, class_name in enumerate(CLASS_NAMES):
        class_dir = os.path.join(data_dir, class_name)
        for fname in sorted(os.listdir(class_dir)):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img = Image.open(os.path.join(class_dir, fname)).convert('RGB')
            img = img.resize((64, 64))
            arr = np.asarray(img, dtype=np.float32).flatten() / 255.0
            images.append(arr)
            labels.append(label_idx)
    return np.array(images), np.array(labels)


def normalize(X_train, X_val, X_test):
    """零均值、单位方差标准化（基于训练集统计量）。"""
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_val - mean) / std, (X_test - mean) / std, mean, std


def stratified_split(X, y, train_ratio=0.7, val_ratio=0.15, seed=42):
    """按类别分层划分 train / val / test。"""
    rng = np.random.RandomState(seed)
    train_idx, val_idx, test_idx = [], [], []
    for c in range(len(CLASS_NAMES)):
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n = len(idx)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        train_idx.extend(idx[:n_train])
        val_idx.extend(idx[n_train:n_train + n_val])
        test_idx.extend(idx[n_train + n_val:])
    train_idx, val_idx, test_idx = map(np.array, [train_idx, val_idx, test_idx])
    return (X[train_idx], y[train_idx],
            X[val_idx], y[val_idx],
            X[test_idx], y[test_idx])


def batch_iterator(X, y, batch_size, shuffle=True):
    """Mini-batch 生成器。"""
    N = len(X)
    idx = np.arange(N)
    if shuffle:
        np.random.shuffle(idx)
    for start in range(0, N, batch_size):
        sel = idx[start:start + batch_size]
        yield X[sel], y[sel]