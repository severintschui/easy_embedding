from pennylane import numpy as np

from sklearn.datasets import make_moons

# random_state = 2
# np.random.seed(random_state)

# 2D Toy thesis_datasets

def crown_gen(n_samples):
    c = [[0,0], [0,0]]
    r = [np.sqrt(.8), np.sqrt(.8-2/np.pi)]
    inputs = []
    targets = []
    dim = 2
    for i in range(n_samples):
        x = 2*(np.random.rand(dim)) - 1
        if np.linalg.norm(x - c[0]) < r[0] and np.linalg.norm(x - c[1]) > r[1]:
            y = 1
        else:
            y = 0
        inputs.append(x)
        targets.append(y)
    return np.array(inputs, requires_grad=False), np.array(targets, requires_grad=False)

def two_waves_gen(n_samples):
    X = np.random.uniform(low=-np.pi, high=np.pi, size=(n_samples, 2))
    y = np.zeros(n_samples, dtype=int)

    f = lambda x: np.sin(2*x) + 1
    g = lambda x: np.sin(3*x + 1) - 1

    for i in range(n_samples):
        if f(X[i, 0]) >= X[i, 1]  >= g(X[i, 0]):
            y[i] = 1

    return np.array(X, requires_grad=False), np.array(y, requires_grad=False)

def checker_board_gen(n_samples):
    inputs = []
    targets = []
    n_x = 3
    n_y = 3
    bounds_x = np.linspace(-2, 2, n_x + 1)
    bounds_y = np.linspace(-2, 2, n_y + 1)
    for i in range(n_samples):
        x = np.random.rand(2) * 4 - 2
        for j in range(n_x):
            for k in range(n_y):
                if x[0] > bounds_x[j] and x[0] <= bounds_x[j + 1] and x[1] > bounds_y[k] and x[1] <= bounds_y[k + 1]:
                    if (j * n_x + k) % 2 == 0:
                        y = 1
                    else:
                        y = 0
        inputs.append(x)
        targets.append(y)
    return np.array(inputs, requires_grad=False), np.array(targets, requires_grad=False)

def wave_gen(n_samples):
    def fun(s):
        return -2 * s + 1.5 * np.sin(1 * np.pi * s)

    inputs = []
    targets = []
    for i in range(n_samples):
        x = 2 * (np.random.rand(2)) - 1
        if x[1] < fun(x[0]): y = 0
        if x[1] >= fun(x[0]): y = 1
        inputs.append(x)
        targets.append(y)
    return np.array(inputs, requires_grad=False), np.array(targets, requires_grad=False)

def spiral_gen(n_samples, noise):
    X = []
    y = []
    for j in range(2):
        for i in range(n_samples//2):
            r = i / n_samples * 2 + np.random.normal(0, scale=noise)/4
            angle = 1.25 * i / n_samples * 2 * np.pi + j*np.pi + np.random.normal(0, scale=noise)
            X.append([r * np.sin(angle), r * np.cos(angle)])
            y.append(j)

    return np.array(X, requires_grad=False), np.array(y, requires_grad=False)

