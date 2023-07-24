import numpy as np
from scipy.linalg import toeplitz


class Policy:
    def __init__(self, theta=None):
        self.theta = theta
        self.is_discrete = False
        self.safe_softplus = lambda x: x*(x>=0) + np.log1p(np.exp(-np.abs(x)))

    def elu(self, inputs, alpha=1.):
        z = inputs
        return np.where(z > 0, z, alpha * (np.exp(z) - 1))

    def linear(self, inputs):
        return inputs

    def relu(self, inputs):
        return inputs * (inputs > 0)

    def set_theta(self, theta):
        self.theta = theta

    def softmax(self, inputs, temperature=1.):
        raise NotImplementedError

        inputs = inputs - max(inputs)
        return relaxed_softmax(inputs, temperature, -1)

    def softplus(self, x):
        return self.safe_softplus(x)

    def tanh(self, inputs):
        return np.tanh(inputs)


class PolicyToeplitz(Policy):
    """See https://github.com/jparkerholder/ASEBO/blob/master/asebo/policies.py"""
    def __init__(self, num_inputs, num_outputs, num_hidden=8, activation='tanh',
                 temperature=1., normalize_output=False, bias=True):
        super().__init__()

        self.policy_name = 'PolicyToeplitz'

        self.policy_params = {}
        self.policy_params['seed'] = 0
        self.policy_params['zeros'] = True
        self.policy_params['ob_dim'] = num_inputs
        self.policy_params['h_dim'] = num_hidden
        self.policy_params['ac_dim'] = num_outputs

        self.activation = activation
        self.temperature = temperature

        self.bias1 = 0
        self.bias2 = 0

        self.act = getattr(self, self.activation)
        self.normalize_output = normalize_output

        self.init_seed = self.policy_params['seed']
        self.ob_dim = self.policy_params['ob_dim']
        self.h_dim = self.policy_params['h_dim']
        self.ac_dim = self.policy_params['ac_dim']
        self.all_weight_init()

        self.set_theta()

    def __call__(self, X):
        h1 = np.dot(self.W1, X) + self.b1
        z1 = self.act(h1)

        h2 = np.dot(self.W2, z1) + self.b2
        z2 = self.act(h2)

        o = np.dot(self.W3, z2)

        if self.normalize_output:
            o = self.softmax(o, self.temperature)
        else:
            o = self.tanh(o)

        return o

    def all_weight_init(self):
        self.w1 = self.weight_init(
            self.ob_dim + self.h_dim -1, self.policy_params['zeros'])
        self.w2 = self.weight_init(
            self.h_dim * 2 - 1, self.policy_params['zeros'])
        self.w3 = self.weight_init(
            self.ac_dim + self.h_dim - 1, self.policy_params['zeros'])

        self.W1 = self._build_layer(
            self.h_dim, self.ob_dim, self.w1)
        self.W2 = self._build_layer(
            self.h_dim, self.h_dim, self.w2)
        self.W3 = self._build_layer(
            self.ac_dim, self.h_dim, self.w3)

        self.b1 = self.weight_init(
            self.h_dim, self.policy_params['zeros'])
        self.b2 = self.weight_init(
            self.h_dim, self.policy_params['zeros'])

        self.theta = np.concatenate(
            [self.w1, self.b1, self.w2, self.b2, self.w3])

        self.d = len(self.theta)

    def get_parametrs(self):
        return self.theta

    def set_theta(self, theta=None):
        if theta is None:
            self.theta = None
            return

        self.all_weight_init()
        self.update(theta)
        self.theta = np.concatenate(
            [self.w1, self.b1, self.w2, self.b2, self.w3])

    def update(self, vec):
        self.w1 += vec[:len(self.w1)]
        vec = vec[len(self.w1):]

        self.b1 += vec[:len(self.b1)]
        vec = vec[len(self.b1):]

        self.w2 += vec[:len(self.w2)]
        vec = vec[len(self.w2):]

        self.b2 += vec[:len(self.b2)]
        vec = vec[len(self.b2):]

        self.w3 += vec

        self.W1 = self._build_layer(self.h_dim, self.ob_dim, self.w1)
        self.W2 = self._build_layer(self.h_dim, self.h_dim, self.w2)
        self.W3 = self._build_layer(self.ac_dim, self.h_dim, self.w3)

    def weight_init(self, d, zeros):
        if zeros:
            return np.zeros(d)

        np.random.seed(self.init_seed)
        return np.random.rand(d) / np.sqrt(d)

    def _build_layer(self, d1, d2, v):
        col = v[:d1]
        row = v[(d1-1):]
        return toeplitz(col, row)
