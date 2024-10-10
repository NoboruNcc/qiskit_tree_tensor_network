from __future__ import annotations

import sys
import pickle
import numpy as np
from collections.abc import Sequence

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from qiskit import QuantumCircuit
from qiskit.primitives import BaseEstimator, Estimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from optimizer import Optimizer, Adam


class PQCTrainerEstimatorQnn:
    """
    パラメータ化量子回路（PQC）のトレーナークラス.EstimatorQNNを使用する.

    Attributes:
        qc_pl (QuantumCircuit): プレースホルダー量子回路
        initial_point (np.ndarray): 初期パラメータ
        optimizer (Optimizer): 最適化アルゴリズム
        estimator (BaseEstimator | None): 量子状態の期待値を推定するためのEstimator
    """

    def __init__(self,
        qc: QuantumCircuit,
        initial_point: Sequence[float],
        optimizer: Optimizer,
        estimator: BaseEstimator | None = None
    ):
        """
        PQCTrainerEstimatorQnnのコンストラクタ.

        Args:
            qc (QuantumCircuit): トレーニングに使用する量子回路
            initial_point (Sequence[float]): 初期パラメータ
            optimizer (Optimizer): 最適化アルゴリズム
            estimator (BaseEstimator | None, optional): Estimatorインスタンス
        """
        self.qc_pl = qc  # placeholder circuit
        self.initial_point = np.array(initial_point)
        self.optimizer = optimizer
        self.estimator = estimator

    def fit(self,
        dataset: Dataset,
        batch_size: int,
        operator: BaseOperator,
        callbacks: list | None = None,
        epochs: int = 1
    ):
        """
        モデルをトレーニングする．

        Args:
            dataset (Dataset): トレーニングデータセット
            batch_size (int): バッチサイズ
            operator (BaseOperator): 測定に使用する演算子
            callbacks (list | None, optional): コールバック関数のリスト
            epochs (int, optional): エポック数

        Returns:
            np.ndarray: 最適化されたパラメータ
        """
        dataloader = DataLoader(dataset, batch_size, shuffle=True, drop_last=True)
        callbacks = callbacks if callbacks is not None else []

        opt_loss = sys.maxsize
        opt_params = None
        params = self.initial_point.copy()

        n_qubits = self.qc_pl.num_qubits
        qnn = EstimatorQNN(
            circuit=self.qc_pl, estimator=self.estimator, observables=operator,
            input_params=self.qc_pl.parameters[:n_qubits],
            weight_params=self.qc_pl.parameters[n_qubits:]
        )
        print(f'num_inputs={qnn.num_inputs}')

        epoch_pbar = tqdm(range(epochs), desc="epochs")
        for epoch in epoch_pbar:
            batch_pbar = tqdm(dataloader, desc=f"epoch {epoch+1}/{epochs}", leave=False)
            for batch, label in batch_pbar:
                batch, label = self._preprocess_batch(batch, label)
                label = label.reshape(label.shape[0], -1)

                expvals = qnn.forward(input_data=batch, weights=params)
                total_loss = np.mean((expvals - label)**2)

                _, grads = qnn.backward(input_data=batch, weights=params)
                grads = np.squeeze(grads, axis=1)
                total_grads = np.mean((expvals - label) * grads, axis=0)

                if total_loss < opt_loss:
                    opt_params = params.copy()
                    opt_loss = total_loss

                    with open('opt_params_iris.pkl', 'wb') as fout:
                        pickle.dump(opt_params, fout)

                self.optimizer.update(params, total_grads)

                for callback in callbacks:
                    callback(total_loss, params)

                # バッチごとにlossを表示
                batch_pbar.set_postfix(loss=f"{total_loss:.4f}")

            # エポックごとにlossを表示
            epoch_pbar.set_postfix(loss=f"{total_loss:.4f}")

        return opt_params

    def _preprocess_batch(self,
        batch: torch.Tensor,
        label: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        バッチデータの前処理を行う．

        Args:
            batch (torch.Tensor): 入力バッチデータ
            label (torch.Tensor): ラベルデータ

        Returns:
            tuple[np.ndarray, np.ndarray]: 前処理されたバッチデータとラベル
        """
        batch = batch.detach().numpy()
        label = label.detach().numpy()
        return batch, label


def RunPQCTrain(
    dataset: Dataset,
    batch_size: int,
    qc: QuantumCircuit,
    operator: BaseOperator,
    init: Sequence[float] | None = None,
    estimator: Estimator | None = None,
    epochs: int = 1,
    interval = 100
):    
    """
    PQCのトレーニングを実行する.

    Args:
        dataset (Dataset): トレーニングデータセット
        batch_size (int): バッチサイズ
        qc (QuantumCircuit): トレーニングに使用する量子回路
        operator (BaseOperator): 測定に使用する演算子
        init (Sequence[float] | None, optional): 初期パラメータ
        estimator (Estimator | None, optional): Estimatorインスタンス
        epochs (int, optional): エポック数
        interval (int, optional): 中間結果を保存する間隔

    Returns:
        tuple[np.ndarray, list]: 最適化されたパラメータと損失の履歴
    """
    # Store intermediate results
    history = {'loss': [], 'params': []}
    cnt = 0

    def store_intermediate_result(loss, params):
        nonlocal cnt
        if cnt % interval != 0:
            return
        history['loss'].append(loss)
        history['params'].append(None)  # とりあえず保存しないことにする。

    # alpha の値はデフォルトより大きいほうが収束が早かった
    optimizer = Adam(alpha=0.01)
    trainer = PQCTrainerEstimatorQnn(
        estimator=estimator, qc=qc, initial_point=init, optimizer=optimizer
    )
    result = trainer.fit(
        dataset, batch_size, operator,
        callbacks=[store_intermediate_result], epochs=epochs
    )

    return result, history['loss']