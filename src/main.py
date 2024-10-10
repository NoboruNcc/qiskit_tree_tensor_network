import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator

from dataset import TransformableDataset, SimpleToTensor, target_transform
from make_init_circuit import make_placeholder_circuit, make_ansatz
from model_wrapper import RunPQCTrain


N_QUBITS = 4
RANDOM_SEED = 41
TEST_SIZE = 0.3
BATCH_SIZE = 32
EPOCHS = 50
INTERVAL = 500

def prepare_data(test_size=TEST_SIZE, random_seed=RANDOM_SEED):
    iris = datasets.load_iris()
    indices = np.where((iris.target==1) | (iris.target==2))
    data = np.squeeze(iris.data[indices, :], axis=0)
    target = iris.target[indices]
    
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=test_size, random_state=random_seed
    )
    
    trainset = TransformableDataset(
        X_train, y_train, SimpleToTensor(float), target_transform
    )
    testset = TransformableDataset(
        X_test, y_test, SimpleToTensor(float), target_transform
    )
    
    return trainset, testset

def prepare_model(n_qubits=N_QUBITS, random_seed=RANDOM_SEED):
    placeholder_circuit = make_placeholder_circuit(n_qubits)
    hamiltonian = SparsePauliOp('IZII')
    length = make_ansatz(n_qubits, dry_run=True)
    
    np.random.seed(random_seed)
    init = np.random.random(length) * 2*math.pi
    
    return placeholder_circuit, hamiltonian, init

def train_model(trainset, placeholder_circuit, hamiltonian, init, epochs=EPOCHS, batch_size=BATCH_SIZE, interval=INTERVAL):
    estimator = Estimator()
    opt_params, loss_list = RunPQCTrain(
        trainset, batch_size,
        placeholder_circuit, hamiltonian, init=init, estimator=estimator,
        epochs=epochs, interval=interval
    )
    return opt_params, loss_list

def plot_loss(loss_list):
    plt.plot(range(len(loss_list)), loss_list)

def main():
    trainset, testset = prepare_data()
    placeholder_circuit, hamiltonian, init = prepare_model()
    opt_params, loss_list = train_model(trainset, placeholder_circuit, hamiltonian, init)
    plot_loss(loss_list)

if __name__ == '__main__':
    main()
