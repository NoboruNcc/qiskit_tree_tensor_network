from __future__ import annotations

import math
import numpy as np


class Optimizer:
    """
    オプティマイザーの基底クラス.
    このクラスは他のオプティマイザーの基礎となり，共通の機能を提供する.

    Attributes:
        hooks (list): パラメータ更新前に実行される関数のリスト.
    """

    def __init__(self):
        """
        Optimizerクラスのコンストラクタ
        hooksリストを初期化する
        """
        self.hooks = []

    def update(self, params, grads) -> None:
        """
        パラメータの更新を行う.
        登録されたすべてのhookを実行した後,update_oneメソッドを呼び出す.

        Args:
            params: 更新対象のパラメータ
            grads: パラメータの勾配
        """
        for f in self.hooks:
            f(params)

        self.update_one(params, grads)

    def update_one(self, params, grads) -> None:
        """
        単一のパラメータとその勾配を用いて更新を行う．
        このメソッドは派生クラスでオーバーライドされる必要がある．

        Args:
            params: 更新対象のパラメータ
            grads: パラメータの勾配

        Raises:
            NotImplementedError: このメソッドが派生クラスで実装されていない場合．
        """
        raise NotImplementedError()

    def add_hook(self, f):
        """
        更新前に実行される関数をhooksリストに追加する.

        Args:
            f: 追加する関数 この関数はパラメータを引数として受け取る
        """
        self.hooks.append(f)


class Adam(Optimizer):
    """
    Adamオプティマイザーの実装.

    Attributes:
        t (int): 更新回数
        alpha (float): 学習率
        beta1 (float): 一次モーメント推定の指数減衰率
        beta2 (float): 二次モーメント推定の指数減衰率
        eps (float): 数値安定性のための小さな値
        ms (dict): 一次モーメントの推定値を格納する辞書
        vs (dict): 二次モーメントの推定値を格納する辞書
    """

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
        """
        パラメータの更新を行う.
        更新回数をインクリメントし,親クラスのupdateメソッドを呼び出す.

        Args:
            *args: 可変長位置引数
            **kwargs: 可変長キーワード引数
        """
        self.t += 1
        super().update(*args, **kwargs)

    @property
    def lr(self):
        """
        現在の学習率を計算する．

        Returns:
            float: 補正された学習率
        """
        fix1 = 1. - math.pow(self.beta1, self.t)
        fix2 = 1. - math.pow(self.beta2, self.t)
        return self.alpha * math.sqrt(fix2) / fix1

    def update_one(self, params, grads):
        """
        単一のパラメータとその勾配を用いて更新を行う.

        Args:
            params (ndarray): 更新対象のパラメータ
            grads (ndarray): パラメータの勾配
        """
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