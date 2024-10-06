from __future__ import annotations

import os
import sys
import math
import pickle
from collections.abc import Sequence
from sklearn import datasets
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt

from optimizer import Adam
from dataset import SimpleToTensor, TransformableDataset, target_transform
from make_init_circuit import make_init_circuit, make_placeholder_circuit, make_ansatz
from model_wrapper import PQCTrainerEstimatorQnn, RunPQCTrain

from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import BaseEstimator, Estimator
from qiskit.quantum_info.operators.base_operator import BaseOperator
from qiskit_machine_learning.neural_networks import EstimatorQNN


iris = datasets.load_iris()

# Sersicolor (1) と Versinica (2)
indices = np.where((iris.target==1) | (iris.target==2))
versicolor_versinica_data = np.squeeze(iris.data[indices, :], axis=0)
versicolor_versinica_target = iris.target[indices]

X_train, X_test, y_train, y_test = train_test_split(
    versicolor_versinica_data, versicolor_versinica_target,
    test_size=0.3, random_state=1234
)

trainset = TransformableDataset(
    X_train, y_train, SimpleToTensor(float), target_transform
)

testset = TransformableDataset(
    X_test, y_test, SimpleToTensor(float), target_transform
)


n_qubits = 4

placeholder_circuit = make_placeholder_circuit(n_qubits)
placeholder_circuit.draw()


hamiltonian = SparsePauliOp('IZII')  # 3rd position from the right, c.f. Fig. 1



length = make_ansatz(n_qubits, dry_run=True)
placeholder_circuit = make_placeholder_circuit(n_qubits)

np.random.seed(10)
init = np.random.random(length) * 2*math.pi

estimator = Estimator()
opt_params, loss_list = RunPQCTrain(
    trainset, 32,
    placeholder_circuit, hamiltonian, init=init, estimator=estimator,
    epochs=100, interval=500)

print(f'final loss={loss_list[-1]}')
print(f'{opt_params=}')


plt.plot(range(len(loss_list)), loss_list)
plt.show()