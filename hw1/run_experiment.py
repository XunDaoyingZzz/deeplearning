"""
一键运行全部实验流程：数据加载 → 超参搜索 → 完整训练 → 测试 → 可视化
"""
import numpy as np
import os
import json

from data_loader import load_data, normalize, stratified_split, CLASS_NAMES
from model import MLP
from train import Trainer
from test import evaluate
from visualize import (plot_training_curves, visualize_weights,
                       plot_confusion_matrix, plot_error_examples)

SAVE_DIR = 'outputs'


def run_search(X_tr, y_tr, X_val, y_val, n_trials=12, epochs=25, seed=42):
    """随机搜索超参数，返回排序后的结果列表和最优参数。"""
    rng = np.random.RandomState(seed)
    results = []
    best_acc, best_params = 0.0, None

    print(f'Random search: {n_trials} trials, {epochs} epochs each\n')

    for i in range(n_trials):
        params = {
            'lr': float(10 ** rng.uniform(-2.5, -1.0)),      # 0.003 ~ 0.1
            'hidden_dim': int(rng.choice([256, 512, 768, 1024])),
            'reg': float(10 ** rng.uniform(-5, -2)),          # 1e-5 ~ 0.01
            'activation': str(rng.choice(['relu', 'tanh'])),
        }
        print(f'[{i + 1}/{n_trials}] {params}')

        model = MLP(X_tr.shape[1], params['hidden_dim'], 10,
                     params['activation'], seed=42)
        trainer = Trainer(model, lr=params['lr'], reg=params['reg'],
                          batch_size=128, lr_decay=0.9, lr_step=8)
        history = trainer.train(X_tr, y_tr, X_val, y_val,
                                epochs=epochs, save_dir=SAVE_DIR, verbose=False)

        # 取该 trial 训练过程中的最优 val_acc
        trial_best_acc = float(max(history['val_acc']))
        trial_best_epoch = int(np.argmax(history['val_acc'])) + 1
        final_train_loss = float(history['train_loss'][-1])
        print(f'  -> best val_acc={trial_best_acc:.4f} (epoch {trial_best_epoch}), '
              f'final train_loss={final_train_loss:.4f}')

        entry = {**params, 'val_acc': trial_best_acc,
                 'best_epoch': trial_best_epoch,
                 'final_train_loss': final_train_loss}
        results.append(entry)

        if trial_best_acc > best_acc:
            best_acc = trial_best_acc
            best_params = params

    results.sort(key=lambda x: x['val_acc'], reverse=True)
    with open(os.path.join(SAVE_DIR, 'search_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nSearch complete. Best: val_acc={best_acc:.4f}')
    print(f'Best params: {best_params}')
    return results, best_params


def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ====== Step 1: 加载数据 ======
    print('=' * 60)
    print('Step 1: Loading data')
    print('=' * 60)
    X, y = load_data('EuroSAT_RGB')
    X_tr, y_tr, X_val, y_val, X_te, y_te = stratified_split(X, y, seed=42)
    X_tr, X_val, X_te, mean, std = normalize(X_tr, X_val, X_te)
    np.savez(os.path.join(SAVE_DIR, 'norm_params.npz'), mean=mean, std=std)
    print(f'Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}')
    print(f'Input dim: {X_tr.shape[1]}  Classes: {len(CLASS_NAMES)}')

    # ====== Step 2: 超参数搜索 ======
    print('\n' + '=' * 60)
    print('Step 2: Hyperparameter search')
    print('=' * 60)
    results, best_params = run_search(X_tr, y_tr, X_val, y_val,
                                       n_trials=12, epochs=25)

    # ====== Step 3: 完整训练 ======
    print('\n' + '=' * 60)
    print('Step 3: Full training with best params (80 epochs)')
    print('=' * 60)
    print(f'Params: {best_params}')
    model = MLP(X_tr.shape[1], best_params['hidden_dim'], 10,
                 best_params['activation'], seed=42)
    trainer = Trainer(model,
                      lr=best_params['lr'],
                      reg=best_params['reg'],
                      batch_size=128,
                      lr_decay=0.7,
                      lr_step=15)
    history = trainer.train(X_tr, y_tr, X_val, y_val,
                            epochs=80, save_dir=SAVE_DIR, verbose=True)

    # ====== Step 4: 测试评估 ======
    print('\n' + '=' * 60)
    print('Step 4: Test evaluation')
    print('=' * 60)
    best_model = MLP.load(os.path.join(SAVE_DIR, 'best_model.npz'))
    acc, cm, y_pred = evaluate(best_model, X_te, y_te)

    # ====== Step 5: 全部可视化 ======
    print('\n' + '=' * 60)
    print('Step 5: Generating visualizations')
    print('=' * 60)
    plot_training_curves(history, save_dir=SAVE_DIR)
    plot_confusion_matrix(cm, save_dir=SAVE_DIR)
    visualize_weights(best_model, save_dir=SAVE_DIR)

    # 错例分析
    wrong_idx = np.where(y_pred != y_te)[0]
    np.random.seed(42)
    np.random.shuffle(wrong_idx)
    plot_error_examples(X_te, y_te, y_pred, wrong_idx[:10],
                        mean=mean, std=std, save_dir=SAVE_DIR)

    print('\n' + '=' * 60)
    print(f'All done! Test accuracy: {acc:.4f}')
    print(f'Outputs saved to {SAVE_DIR}/')
    print('=' * 60)


if __name__ == '__main__':
    main()