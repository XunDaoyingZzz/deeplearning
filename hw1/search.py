import numpy as np
import itertools
import json
import os
from model import MLP
from train import Trainer


def grid_search(X_train, y_train, X_val, y_val,
                param_grid=None, epochs=30, save_dir='outputs'):
    """网格搜索超参数。

    Args:
        param_grid: dict，如 {'lr': [0.1, 0.01], 'hidden_dim': [256, 512], ...}
    """
    if param_grid is None:
        param_grid = {
            'lr': [0.1, 0.01, 0.001],
            'hidden_dim': [128, 256, 512],
            'reg': [1e-3, 1e-4, 1e-5],
            'activation': ['relu', 'tanh'],
        }

    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    print(f'Grid search: {len(combos)} combinations\n')

    results = []
    best_acc, best_params = 0.0, None

    for i, vals in enumerate(combos):
        params = dict(zip(keys, vals))
        print(f'[{i + 1}/{len(combos)}] {params}')

        model = MLP(input_dim=X_train.shape[1],
                     hidden_dim=params['hidden_dim'],
                     output_dim=10,
                     activation=params['activation'],
                     seed=42)
        trainer = Trainer(model, lr=params['lr'], reg=params['reg'])
        trainer.train(X_train, y_train, X_val, y_val,
                      epochs=epochs, save_dir=save_dir, verbose=False)

        val_loss, val_acc = trainer.evaluate(X_val, y_val)
        print(f'  -> val_acc={val_acc:.4f}')

        entry = {**params, 'val_acc': float(val_acc), 'val_loss': float(val_loss)}
        results.append(entry)

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            model.save(os.path.join(save_dir, 'best_model.npz'))

    results.sort(key=lambda x: x['val_acc'], reverse=True)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'search_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nBest params: {best_params}  val_acc={best_acc:.4f}')
    return results, best_params


def random_search(X_train, y_train, X_val, y_val,
                  n_trials=20, epochs=30, save_dir='outputs', seed=42):
    """随机搜索超参数。"""
    rng = np.random.RandomState(seed)
    results = []
    best_acc, best_params = 0.0, None

    print(f'Random search: {n_trials} trials\n')

    for i in range(n_trials):
        params = {
            'lr': float(10 ** rng.uniform(-3, -0.5)),
            'hidden_dim': int(rng.choice([128, 256, 512, 1024])),
            'reg': float(10 ** rng.uniform(-5, -2)),
            'activation': str(rng.choice(['relu', 'tanh'])),
        }
        print(f'[{i + 1}/{n_trials}] {params}')

        model = MLP(input_dim=X_train.shape[1],
                     hidden_dim=params['hidden_dim'],
                     output_dim=10,
                     activation=params['activation'],
                     seed=42)
        trainer = Trainer(model, lr=params['lr'], reg=params['reg'])
        trainer.train(X_train, y_train, X_val, y_val,
                      epochs=epochs, save_dir=save_dir, verbose=False)

        val_loss, val_acc = trainer.evaluate(X_val, y_val)
        print(f'  -> val_acc={val_acc:.4f}')

        entry = {**params, 'val_acc': float(val_acc), 'val_loss': float(val_loss)}
        results.append(entry)

        if val_acc > best_acc:
            best_acc = val_acc
            best_params = params
            model.save(os.path.join(save_dir, 'best_model.npz'))

    results.sort(key=lambda x: x['val_acc'], reverse=True)
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'search_results.json'), 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\nBest params: {best_params}  val_acc={best_acc:.4f}')
    return results, best_params