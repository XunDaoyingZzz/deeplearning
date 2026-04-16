import numpy as np
import os
from data_loader import batch_iterator


class Trainer:
    """SGD 训练器，支持学习率衰减、L2 正则化、最优模型保存。"""

    def __init__(self, model, lr=0.01, lr_decay=0.95, lr_step=10,
                 reg=1e-4, batch_size=128):
        self.model = model
        self.lr_init = lr
        self.lr = lr
        self.lr_decay = lr_decay   # 每 lr_step 个 epoch 衰减一次
        self.lr_step = lr_step
        self.reg = reg
        self.batch_size = batch_size
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': []}

    # 交叉熵损失（含 L2 正则项）
    def compute_loss(self, probs, y):
        N = len(y)
        data_loss = -np.mean(np.log(probs[np.arange(N), y] + 1e-12))
        reg_loss = 0.5 * self.reg * (np.sum(self.model.W1 ** 2) +
                                      np.sum(self.model.W2 ** 2))
        return data_loss + reg_loss

    # SGD 参数更新
    def _sgd_step(self, grads):
        self.model.W1 -= self.lr * grads['W1']
        self.model.b1 -= self.lr * grads['b1']
        self.model.W2 -= self.lr * grads['W2']
        self.model.b2 -= self.lr * grads['b2']

    # 学习率衰减（step decay）
    def _update_lr(self, epoch):
        self.lr = self.lr_init * (self.lr_decay ** (epoch // self.lr_step))

    # 评估准确率与损失
    def evaluate(self, X, y):
        probs = self.model.forward(X)
        loss = self.compute_loss(probs, y)
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y)
        return loss, acc

    # Training loop
    def train(self, X_train, y_train, X_val, y_val,
              epochs=50, save_dir='outputs', verbose=True):
        os.makedirs(save_dir, exist_ok=True)
        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            self._update_lr(epoch)

            # 训练一个 epoch
            epoch_losses = []
            for X_batch, y_batch in batch_iterator(X_train, y_train,
                                                   self.batch_size, shuffle=True):
                probs = self.model.forward(X_batch)
                loss = self.compute_loss(probs, y_batch)
                epoch_losses.append(loss)
                grads = self.model.backward(y_batch, reg=self.reg)
                self._sgd_step(grads)

            train_loss = np.mean(epoch_losses)
            val_loss, val_acc = self.evaluate(X_val, y_val)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 保存最优模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                self.model.save(os.path.join(save_dir, 'best_model.npz'))

            if verbose:
                print(f"Epoch {epoch:3d}/{epochs} | "
                      f"lr={self.lr:.6f} | "
                      f"train_loss={train_loss:.4f} | "
                      f"val_loss={val_loss:.4f} | "
                      f"val_acc={val_acc:.4f}"
                      f"{' *' if val_acc >= best_val_acc else ''}")

        # 保存训练历史
        np.savez(os.path.join(save_dir, 'history.npz'), **self.history)
        print(f"\nTraining complete. Best val acc: {best_val_acc:.4f}")
        return self.history