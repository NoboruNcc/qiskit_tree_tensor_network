from __future__ import annotations

import math
import numpy as np


class Optimizer:
    def __init__(self):
        self.hooks = []

    def update(self, params, grads) -> None:
        for f in self.hooks:
            f(params)

        self.update_one(params, grads)

    def update_one(self, params, grads) -> None:
        raise NotImplementedError()

    def add_hook(self, f):
        self.hooks.append(f)


class Adam(Optimizer):
    def __init__(self, alpha=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        super().__init__()
        self.t = 0
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.ms = {}
        self.vs = {}

    def update(self, *args, **kwargs):
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, params, grads):
        key = 'Adam'
        if key not in self.ms:
            self.ms[key] = np.zeros_like(params)
            self.vs[key] = np.zeros_like(params)

        m, v = self.ms[key], self.vs[key]
        beta1, beta2, eps = self.beta1, self.beta2, self.eps
        grad = grads

        m += (1 - beta1) * (grad - m)
        v += (1 - beta2) * (grad * grad - v)
        params -= self.lr * m / (np.sqrt(v) + eps)