"""
HW1: 从零构建三层 MLP 分类器，实现 EuroSAT 地表覆盖图像分类。
用法:
    python main.py train   [--hidden_dim 512] [--lr 0.01] [--epochs 50] ...
    python main.py test    [--model_path outputs/best_model.npz]
    python main.py search  [--search_type grid] [--n_trials 20]
    python main.py visualize
"""
import argparse
import numpy as np
import os

from data_loader import load_data, normalize, stratified_split
from model import MLP
from train import Trainer
from test import evaluate, find_misclassified
from search import grid_search, random_search
from visualize import (plot_training_curves, visualize_weights,
                       plot_confusion_matrix, plot_error_examples)

SAVE_DIR = 'outputs'


def prepare_data(data_dir='EuroSAT_RGB', seed=42):
    """加载并预处理数据，返回 train/val/test 集和标准化参数。"""
    print('Loading data...')
    X, y = load_data(data_dir)
    print(f'  Total: {len(X)} images, input dim={X.shape[1]}')

    X_tr, y_tr, X_val, y_val, X_te, y_te = stratified_split(X, y, seed=seed)
    print(f'  Train: {len(X_tr)}  Val: {len(X_val)}  Test: {len(X_te)}')

    X_tr, X_val, X_te, mean, std = normalize(X_tr, X_val, X_te)
    np.savez(os.path.join(SAVE_DIR, 'norm_params.npz'), mean=mean, std=std)
    return X_tr, y_tr, X_val, y_val, X_te, y_te, mean, std


def cmd_train(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    X_tr, y_tr, X_val, y_val, *_ = prepare_data(args.data_dir, args.seed)

    model = MLP(input_dim=X_tr.shape[1],
                hidden_dim=args.hidden_dim,
                output_dim=10,
                activation=args.activation,
                seed=args.seed)

    trainer = Trainer(model,
                      lr=args.lr,
                      lr_decay=args.lr_decay,
                      lr_step=args.lr_step,
                      reg=args.reg,
                      batch_size=args.batch_size)

    history = trainer.train(X_tr, y_tr, X_val, y_val,
                            epochs=args.epochs, save_dir=SAVE_DIR)
    plot_training_curves(history, save_dir=SAVE_DIR)


def cmd_test(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    *_, X_te, y_te, mean, std = prepare_data(args.data_dir, args.seed)

    model = MLP.load(args.model_path)
    print(f'\nLoaded model from {args.model_path}')
    print(f'  hidden_dim={model.hidden_dim}, activation={model.act_name}\n')

    acc, cm, y_pred = evaluate(model, X_te, y_te)
    plot_confusion_matrix(cm, save_dir=SAVE_DIR)

    # 错例分析
    wrong_idx = np.where(y_pred != y_te)[0]
    np.random.seed(args.seed)
    np.random.shuffle(wrong_idx)
    sel = wrong_idx[:10]
    plot_error_examples(X_te, y_te, y_pred, sel, mean=mean, std=std,
                        save_dir=SAVE_DIR)


def cmd_search(args):
    os.makedirs(SAVE_DIR, exist_ok=True)
    X_tr, y_tr, X_val, y_val, *_ = prepare_data(args.data_dir, args.seed)

    if args.search_type == 'grid':
        grid_search(X_tr, y_tr, X_val, y_val,
                    epochs=args.search_epochs, save_dir=SAVE_DIR)
    else:
        random_search(X_tr, y_tr, X_val, y_val,
                      n_trials=args.n_trials,
                      epochs=args.search_epochs, save_dir=SAVE_DIR)


def cmd_visualize(args):
    os.makedirs(SAVE_DIR, exist_ok=True)

    # 训练曲线
    hist_path = os.path.join(SAVE_DIR, 'history.npz')
    if os.path.exists(hist_path):
        h = dict(np.load(hist_path))
        plot_training_curves(h, save_dir=SAVE_DIR)

    # 权重可视化
    model = MLP.load(args.model_path)
    visualize_weights(model, save_dir=SAVE_DIR)


def main():
    parser = argparse.ArgumentParser(description='HW1: 3-layer MLP on EuroSAT')
    sub = parser.add_subparsers(dest='command')

    # ---- train ----
    p_train = sub.add_parser('train')
    p_train.add_argument('--data_dir', default='EuroSAT_RGB')
    p_train.add_argument('--hidden_dim', type=int, default=512)
    p_train.add_argument('--activation', default='relu', choices=['relu', 'sigmoid', 'tanh'])
    p_train.add_argument('--lr', type=float, default=0.01)
    p_train.add_argument('--lr_decay', type=float, default=0.95)
    p_train.add_argument('--lr_step', type=int, default=10)
    p_train.add_argument('--reg', type=float, default=1e-4)
    p_train.add_argument('--batch_size', type=int, default=128)
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--seed', type=int, default=42)

    # ---- test ----
    p_test = sub.add_parser('test')
    p_test.add_argument('--data_dir', default='EuroSAT_RGB')
    p_test.add_argument('--model_path', default=os.path.join(SAVE_DIR, 'best_model.npz'))
    p_test.add_argument('--seed', type=int, default=42)

    # ---- search ----
    p_search = sub.add_parser('search')
    p_search.add_argument('--data_dir', default='EuroSAT_RGB')
    p_search.add_argument('--search_type', default='random', choices=['grid', 'random'])
    p_search.add_argument('--n_trials', type=int, default=20)
    p_search.add_argument('--search_epochs', type=int, default=30)
    p_search.add_argument('--seed', type=int, default=42)

    # ---- visualize ----
    p_vis = sub.add_parser('visualize')
    p_vis.add_argument('--model_path', default=os.path.join(SAVE_DIR, 'best_model.npz'))

    args = parser.parse_args()

    if args.command == 'train':
        cmd_train(args)
    elif args.command == 'test':
        cmd_test(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'visualize':
        cmd_visualize(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()