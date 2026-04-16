import numpy as np
from data_loader import CLASS_NAMES


def confusion_matrix(y_true, y_pred, num_classes=10):
    """计算混淆矩阵 (真实标签 × 预测标签)。"""
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def print_confusion_matrix(cm, class_names=CLASS_NAMES):
    """格式化打印混淆矩阵。"""
    max_len = max(len(n) for n in class_names)
    header = ' ' * (max_len + 2) + '  '.join(f'{n[:6]:>6s}' for n in class_names)
    print(header)
    for i, name in enumerate(class_names):
        row = '  '.join(f'{cm[i, j]:6d}' for j in range(len(class_names)))
        print(f'{name:>{max_len}s}  {row}')


def evaluate(model, X_test, y_test):
    """在测试集上评估模型，返回准确率、混淆矩阵、预测结果。"""
    y_pred = model.predict(X_test)
    acc = np.mean(y_pred == y_test)
    cm = confusion_matrix(y_test, y_pred)

    print(f'Test Accuracy: {acc:.4f} ({np.sum(y_pred == y_test)}/{len(y_test)})')
    print('\nConfusion Matrix (rows=true, cols=pred):')
    print_confusion_matrix(cm)

    # 各类别准确率
    print('\nPer-class accuracy:')
    for i, name in enumerate(CLASS_NAMES):
        total = cm[i].sum()
        correct = cm[i, i]
        print(f'  {name:>25s}: {correct}/{total}  ({correct / total:.2%})')

    return acc, cm, y_pred


def find_misclassified(model, X_test, y_test, max_samples=10):
    """找出分类错误的样本索引，用于错例分析。"""
    y_pred = model.predict(X_test)
    wrong_idx = np.where(y_pred != y_test)[0]
    np.random.shuffle(wrong_idx)
    sel = wrong_idx[:max_samples]
    return sel, y_test[sel], y_pred[sel]