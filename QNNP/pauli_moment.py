import sys
import pennylane as qml
from QMLatomLoader import *
from tqdm import tqdm
import torch
from torch.utils.tensorboard import SummaryWriter

epochs = 1000
batchs = 20

para = np.array([[0, 0.5],
                 [0.1, 1],
                 [1, 5]])

n_qubit = para.shape[0]

dev = qml.device("default.qubit", wires=para.shape[0])


@qml.qnode(dev)
def atomicoutput(descript, H):
    """The energy expectation value of a Hamiltonian"""
    for i in range(n_qubit):
        qml.Rot(-descript[i, 1], descript[i, 0], descript[i, 1], wires=i)

    return qml.expval(H)


def hamiltonian(parameters, descriptor_size):
    # print(descriptor_size[:, np.newaxis])
    coeff = np.zeros((n_qubit, 4)) + descriptor_size[:, np.newaxis]
    coeff = coeff.flatten()
    coeff = coeff * parameters
    obs = []
    for i in range(n_qubit):
        obs.append(qml.Identity(i))
        obs.append(qml.PauliX(i))
        obs.append(qml.PauliY(i))
        obs.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeff, obs)


# def losses(param,ground_energy,descriptor,descript_sizes,n_atom):

#     outputs=0
#     for n in range(n_atom):     
#         outputs+=atomicoutput(descriptor[n],hamiltonian(param,descript_sizes[n]))
#     return ((ground_energy-outputs)/n_atom)**2

class NeuralNet:
    def __init__(self, shape: list, activation_function:str='ReLu'):
        """
        Args:
            shape: array like object to specify NN shape. i.e. [2, 3, 2] represent len 2 input and output layer and one
            len 3 hidden layer. must be longer than 2.
            activation_function: Str or Callable. which activation function to use
        """
        assert len(shape) >= 2
        self.shape = shape
        if isinstance(activation_function, str):
            if activation_function=='Sigmoid':
                self.activation_function = self.sigmoid
            elif activation_function=='RelU':
                self.activation_function = self.relu
            else:
                raise NotImplementedError(f'activation function {activation_function} is unknown')
        elif isinstance(activation_function, callable):
            self.activation_function = activation_function
        else:
            raise TypeError('Invalid type of activation function')

    def initial_parameters(self):
        # initialize parameters (weight, bias)
        weights = []
        biases = []
        for i in range(len(self.shape)-1):
            weights.append(self.generate_wt(self.shape[i], self.shape[i+1]))
            biases.append(self.generate_b(self.shape[i+1]))
        return weights, biases


    # activation function

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def relu(x):
        return np.where(x > 0, x, 0)

    # Creating the Feed forward neural network
    # 1 Input layer(1, 30)
    # 1 hidden layer (1, 5)
    # 1 output layer(3, 3)

    def forward(self, x, weights, biases):
        """
        Args:
            x: input
            weights, biases: parameters

        Returns:
        """
        for l in range(len(self.shape)-1):
            x = x.dot(weights[l])+biases[l]
        return x

    # initializing the weights randomly
    @staticmethod
    def generate_wt(x, y):
        l = []
        for i in range(x * y):
            l.append(np.random.randn())
        return np.array(l).reshape(x, y)

    @staticmethod
    def generate_b(x):
        l = []
        for i in range(x):
            l.append(np.random.randn())
        return np.array(l)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    init_params = np.random.random(n_qubit * 4, requires_grad=True)
    opt = qml.AdagradOptimizer(stepsize=0.3)
    params = init_params
    eta_val = NeuralNet(shape=[3, 4, 4, 1])
    nn_weights, nn_biases = eta_val.initial_parameters()

    loss_list = []
    writer = SummaryWriter()
    if batchs == 1:
        loadatom = AtomLoader1(
            sampler='random',
            numb=epochs,
            # classic=True,
            new_parameter=eta_val,
            classic_parameter=para,
            weigthed=True
        )
        for i in tqdm(range(epochs)):
            ground_energy = loadatom[i]['ground_energy']
            descriptor = loadatom[i]['descriptor']
            descript_sizes = loadatom[i]['descriptor_size']
            n_atom = loadatom[i]['atomic_number']
            # descriptor.requires_grad = False
            descriptor.requires_grad = True
            descript_sizes.requires_grad = False


            def losses(param, temp):
                outputs = 0
                for n in range(n_atom):
                    outputs += atomicoutput(descriptor[n], hamiltonian(param, descript_sizes[n]))
                return np.sqrt(((ground_energy - outputs) / n_atom) ** 2)

            (params, eta_val), loss = opt.step_and_cost(losses, params, eta_val)
            loss_list.append(loss)
            writer.add_scalar('Loss(batches=1)', loss, i)
    else:
        loadatom = AtomLoader2(
            sampler='random',
            epochs=epochs,
            batchs=batchs,
            # classic=True,
            new_parameter=eta_val,
            classic_parameter=para,
            weigthed=True
        )
        for i in tqdm(range(epochs)):
            def losses(param, temp):
                loss = 0
                x = loadatom[i]
                for j in range(batchs):
                    ground_energy = x[j]['ground_energy']
                    descriptor = x[j]['descriptor']
                    descript_sizes = x[j]['descriptor_size']
                    n_atom = x[j]['atomic_number']
                    # descriptor.requires_grad = False
                    descriptor.requires_grad = True
                    descript_sizes.requires_grad = False
                    out = 0
                    for n in range(n_atom):
                        out += atomicoutput(descriptor[n], hamiltonian(param, descript_sizes[n]))
                    loss += np.sqrt(((ground_energy - out) / n_atom) ** 2)
                return loss / batchs


            (params, eta_val), loss = opt.step_and_cost(losses, params, eta_val)
            loss_list.append(loss)
            writer.add_scalar(f'Loss(batches={batchs})', loss, i)
