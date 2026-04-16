import numpy as np
import matplotlib.pyplot as plt
import os
from data_loader import CLASS_NAMES


def plot_training_curves(history, save_dir='outputs'):
    """绘制 train/val loss 曲线和 val accuracy 曲线。"""
    epochs = range(1, len(history['train_loss']) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Loss 曲线
    ax1.plot(epochs, history['train_loss'], label='Train Loss')
    ax1.plot(epochs, history['val_loss'], label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True)

    # Accuracy 曲线
    ax2.plot(epochs, history['val_acc'], label='Val Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'training_curves.png'), dpi=150)
    plt.close()
    print(f'Saved training_curves.png')


def visualize_weights(model, save_dir='outputs'):
    """将第一层权重 reshape 成 (64,64,3) 图像并可视化。"""
    W = model.W1  # (12288, H)
    H = W.shape[1]
    ncols = min(8, H)
    nrows = min(4, (H + ncols - 1) // ncols)
    n_show = nrows * ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(2 * ncols, 2 * nrows))
    axes = np.array(axes).flatten()

    for i in range(n_show):
        w = W[:, i].reshape(64, 64, 3)
        # 归一化到 [0,1] 用于显示
        w = (w - w.min()) / (w.max() - w.min() + 1e-8)
        axes[i].imshow(w)
        axes[i].set_title(f'#{i}', fontsize=8)
        axes[i].axis('off')

    plt.suptitle('First Hidden Layer Weights', fontsize=14)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'weight_visualization.png'), dpi=150)
    plt.close()
    print(f'Saved weight_visualization.png')


def plot_confusion_matrix(cm, save_dir='outputs'):
    """绘制混淆矩阵热力图。"""
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(xticks=range(len(CLASS_NAMES)),
           yticks=range(len(CLASS_NAMES)),
           xticklabels=CLASS_NAMES,
           yticklabels=CLASS_NAMES,
           ylabel='True Label',
           xlabel='Predicted Label',
           title='Confusion Matrix')

    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # 在每个格子里标注数字
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha='center', va='center',
                    color='white' if cm[i, j] > thresh else 'black',
                    fontsize=7)

    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f'Saved confusion_matrix.png')


def plot_error_examples(X_test, y_true, y_pred, indices,
                        mean=None, std=None, save_dir='outputs'):
    """可视化分类错误的样本。"""
    n = len(indices)
    fig, axes = plt.subplots(2, min(5, n), figsize=(3 * min(5, n), 7))
    if n <= 5:
        axes = axes.reshape(2, -1) if n > 1 else axes.reshape(2, 1)

    for row_start, row_axes in enumerate([axes[0], axes[1]]):
        for col, ax in enumerate(row_axes):
            idx_pos = row_start * len(row_axes) + col
            if idx_pos >= n:
                ax.axis('off')
                continue
            idx = indices[idx_pos]
            img = X_test[idx].copy()
            if mean is not None and std is not None:
                img = img * std + mean
            img = img.reshape(64, 64, 3).clip(0, 1)
            ax.imshow(img)
            ax.set_title(f'True: {CLASS_NAMES[y_true[idx]]}\n'
                         f'Pred: {CLASS_NAMES[y_pred[idx]]}', fontsize=8)
            ax.axis('off')

    plt.suptitle('Misclassified Examples', fontsize=14)
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'error_analysis.png'), dpi=150)
    plt.close()
    print(f'Saved error_analysis.png')