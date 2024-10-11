import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from qiskit.quantum_info import SparsePauliOp
from qiskit.primitives import Estimator
from torch.utils.data import DataLoader
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

def prepare_model(n_qubits=N_QUBITS, random_seed=RANDOM_SEED, save_fig=None):
    placeholder_circuit = make_placeholder_circuit(n_qubits)
    if save_fig:
        placeholder_circuit.draw(output='mpl', filename=save_fig)
        plt.close()
    hamiltonian = SparsePauliOp('IZII')
    length = make_ansatz(n_qubits, dry_run=True)
    
    np.random.seed(random_seed)
    init = np.random.random(length) * 2*math.pi
    
    return placeholder_circuit, hamiltonian, init

def train_model(trainset, placeholder_circuit, hamiltonian, init, save_path=None, epochs=EPOCHS, batch_size=BATCH_SIZE, interval=INTERVAL):
    estimator = Estimator()
    opt_params, loss_list = RunPQCTrain(
        trainset, batch_size,
        placeholder_circuit, hamiltonian, init=init, estimator=estimator,
        epochs=epochs, interval=interval, save_path=save_path
    )
    return opt_params, loss_list

def plot_loss(loss_list, save_path=None):
    plt.plot(range(len(loss_list)), loss_list)
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()



def validate_model(testset, hamiltonian, opt_params, save_path=None):
    testloader = DataLoader(testset, 32)

    qc_pl = make_placeholder_circuit(N_QUBITS)
    estimator = Estimator()

    total = 0
    total_correct = 0

    for i, (batch, label) in enumerate(testloader):
        batch, label = batch.detach().numpy(), label.detach().numpy()

        qc_list = []

        for i in range(batch.shape[0]):
            data = batch[i, :]
        
            qc_placeholder = qc_pl.copy()
            qc = qc_placeholder.assign_parameters(data.tolist() + opt_params.tolist())
            qc_list.append(qc)

        job = estimator.run(qc_list, [hamiltonian]*len(qc_list))
        result = job.result()
        expvals = result.values

        predict_labels = np.ones_like(expvals)
        predict_labels[np.where(expvals < 0)] = -1
        predict_labels = predict_labels.astype(int)

        total_correct += np.sum(predict_labels == label)
        total += batch.shape[0]

    print(f'test acc={np.round(total_correct/total, 2)}')
    if save_path:
        with open(save_path, 'w') as f:
            f.write('test_acc\n')
            f.write(f'{np.round(total_correct/total, 2)}\n')

def main():
    trainset, testset = prepare_data()
    placeholder_circuit, hamiltonian, init = prepare_model(save_fig='results/placeholder_circuit.png')
    opt_params, loss_list = train_model(trainset=trainset, placeholder_circuit=placeholder_circuit, hamiltonian=hamiltonian, init=init, save_path='results/opt_params_iris.pkl')
    plot_loss(loss_list, save_path='results/loss.png')
    validate_model(testset, hamiltonian, opt_params, save_path='results/test_accuracy.csv')


if __name__ == '__main__':
    main()
