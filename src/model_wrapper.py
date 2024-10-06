from __future__ import annotations

import sys
import pickle
from collections.abc import Sequence
import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimator, Estimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN
from optimizer import Optimizer, Adam


class PQCTrainerEstimatorQnn:
    def __init__(self,
        qc: QuantumCircuit,
        initial_point: Sequence[float],
        optimizer: Optimizer,
        estimator: BaseEstimator | None = None
    ):
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

        for epoch in range(epochs):
            for batch, label in dataloader:
                batch, label = self._preprocess_batch(batch, label)
                label = label.reshape(label.shape[0], -1)

                expvals = qnn.forward(input_data=batch, weights=params)
                total_loss = np.mean((expvals - label)**2)

                _, grads = qnn.backward(input_data=batch, weights=params)
                grads = np.squeeze(grads, axis=1)
                # コスト関数の勾配を組み立てて、バッチでの平均をとる。
                total_grads = np.mean((expvals - label) * grads, axis=0)

                if total_loss < opt_loss:
                    opt_params = params.copy()
                    opt_loss = total_loss

                    with open('opt_params_iris.pkl', 'wb') as fout:
                        pickle.dump(opt_params, fout)

                self.optimizer.update(params, total_grads)

                for callback in callbacks:
                    callback(total_loss, params)

        return opt_params

    def _preprocess_batch(self,
        batch: torch.Tensor,
        label: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray]:
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
    # Store intermediate results
    history = {'loss': [], 'params': []}
    cnt = 0

    def store_intermediate_result(loss, params):
        nonlocal cnt
        if cnt % interval != 0:
            return
        history['loss'].append(loss)
        history['params'].append(None)  # とりあえず保存しないことにする。
        print(f'{loss=}')

    # alpha の値はデフォルトより大きいほうが収束が早かった。
    optimizer = Adam(alpha=0.01)
    trainer = PQCTrainerEstimatorQnn(
        estimator=estimator, qc=qc, initial_point=init, optimizer=optimizer
    )
    result = trainer.fit(
        dataset, batch_size, operator,
        callbacks=[store_intermediate_result], epochs=epochs
    )

    return result, history['loss']