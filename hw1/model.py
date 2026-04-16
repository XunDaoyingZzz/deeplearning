import numpy as np


# 激活函数
class ReLU:
    @staticmethod
    def forward(z):
        return np.maximum(0, z)

    @staticmethod
    def backward(z):
        return (z > 0).astype(z.dtype)


class Sigmoid:
    @staticmethod
    def forward(z):
        return np.where(z >= 0,
                        1.0 / (1.0 + np.exp(-z)),
                        np.exp(z) / (1.0 + np.exp(z)))

    @staticmethod
    def backward(z):
        s = Sigmoid.forward(z)
        return s * (1. - s)


class Tanh:
    @staticmethod
    def forward(z):
        return np.tanh(z)

    @staticmethod
    def backward(z):
        return 1.0 - np.tanh(z) ** 2


ACTIVATIONS = {'relu': ReLU, 'sigmoid': Sigmoid, 'tanh': Tanh}


def softmax(z):
    z_shifted = z - z.max(axis=1, keepdims=True)
    e = np.exp(z_shifted)
    return e / e.sum(axis=1, keepdims=True)


# 三层 MLP(Input → Hidden → Output)
class MLP:
    """
    三层神经网络：输入层 → 隐藏层 → 输出层(softmax)
    手动实现前向传播与反向传播。
    """

    def __init__(self, input_dim, hidden_dim, output_dim, activation='relu', seed=None):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.act_name = activation
        self.act = ACTIVATIONS[activation]
        self._init_weights(seed)
        self.cache = {}

    # 初始化权重
    def _init_weights(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        if self.act_name == 'relu':
            # He初始化
            scale1 = np.sqrt(2.0 / self.input_dim)
            scale2 = np.sqrt(2.0 / self.hidden_dim)
        else:
            # Xavier初始化
            scale1 = np.sqrt(1.0 / self.input_dim)
            scale2 = np.sqrt(1.0 / self.hidden_dim)
        self.W1 = (np.random.randn(self.input_dim, self.hidden_dim) * scale1).astype(np.float32)
        self.b1 = np.zeros(self.hidden_dim, dtype=np.float32)
        self.W2 = (np.random.randn(self.hidden_dim, self.output_dim) * scale2).astype(np.float32)
        self.b2 = np.zeros(self.output_dim, dtype=np.float32)

    # 前向传播
    def forward(self, X):
        z1 = X @ self.W1 + self.b1          # (N, H)
        a1 = self.act.forward(z1)            # (N, H)
        z2 = a1 @ self.W2 + self.b2         # (N, C)
        probs = softmax(z2)                  # (N, C)
        self.cache = dict(X=X, z1=z1, a1=a1, probs=probs)
        return probs

    # 反向传播
    def backward(self, y_true, reg=0.0):
        X, z1, a1, probs = (self.cache[k] for k in ('X', 'z1', 'a1', 'probs'))
        N = X.shape[0]

        # one-hot
        y_oh = np.zeros_like(probs)
        y_oh[np.arange(N), y_true] = 1.0

        # 输出层梯度  (softmax + cross-entropy 的联合导数)
        dz2 = (probs - y_oh) / N             # (N, C)
        dW2 = a1.T @ dz2 + reg * self.W2     # (H, C)
        db2 = dz2.sum(axis=0)                # (C,)

        # 隐藏层梯度
        da1 = dz2 @ self.W2.T                # (N, H)
        dz1 = da1 * self.act.backward(z1)    # (N, H)
        dW1 = X.T @ dz1 + reg * self.W1      # (D, H)
        db1 = dz1.sum(axis=0)                # (H,)

        return {'W1': dW1, 'b1': db1, 'W2': dW2, 'b2': db2}

    # 预测
    def predict(self, X):
        return np.argmax(self.forward(X), axis=1)

    # 保存 / 加载
    def save(self, path):
        np.savez(path,
                 W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2,
                 config=[self.input_dim, self.hidden_dim, self.output_dim],
                 activation=self.act_name)

    @classmethod
    def load(cls, path):
        d = np.load(path, allow_pickle=True)
        cfg = d['config']
        m = cls(int(cfg[0]), int(cfg[1]), int(cfg[2]), str(d['activation']))
        m.W1, m.b1, m.W2, m.b2 = d['W1'], d['b1'], d['W2'], d['b2']
        return m