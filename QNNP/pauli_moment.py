import sys
import pennylane as qml
# import numpy as np
from pennylane import numpy as np
from QMLatomLoader import *
from tqdm import tqdm
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

    def initialize_parameters(self):
        # initialize parameters (weight, bias)
        parameters = []
        for i in range(len(self.shape)-1):
            parameters.append((np.random.random(size=(self.shape[i], self.shape[i+1])), np.random.random(size=self.shape[i])))

        return parameters


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

    def forward(self, x, parameters):
        """
        Args:
            x: input
            parameters: tuple of parameters

        Returns:
        """


    def f_forward(self, x, w1, w2):
        # hidden
        z1 = x.dot(w1)  # input from layer 1
        a1 = self.sigmoid(z1)  # out put of layer 2

        # Output layer
        z2 = a1.dot(w2)  # input of out layer
        a2 = self.sigmoid(z2)  # output of out layer
        return a2

    # initializing the weights randomly
    def generate_wt(self, x, y):
        l = []
        for i in range(x * y):
            l.append(np.random.randn())
        return (np.array(l).reshape(x, y))

    # for loss we will be using mean square error(MSE)
    def loss(self, out, Y):
        s = (np.square(out - Y))
        s = np.sum(s) / len(y)
        return (s)

    # Back propagation of error
    def back_prop(self, x, y, w1, w2, alpha):

        # hidden layer
        z1 = x.dot(w1)  # input from layer 1
        a1 = self.sigmoid(z1)  # output of layer 2

        # Output layer
        z2 = a1.dot(w2)  # input of out layer
        a2 = self.sigmoid(z2)  # output of out layer
        # error in output layer
        d2 = (a2 - y)
        d1 = np.multiply((w2.dot((d2.transpose()))).transpose(),
                         (np.multiply(a1, 1 - a1)))

        # Gradient for w1 and w2
        w1_adj = x.transpose().dot(d1)
        w2_adj = a1.transpose().dot(d2)

        # Updating parameters
        w1 = w1 - (alpha * (w1_adj))
        w2 = w2 - (alpha * (w2_adj))

        return (w1, w2)

    def train(self, x, Y, w1, w2, alpha=0.01, epoch=10):
        acc = []
        losss = []
        for j in range(epoch):
            l = []
            for i in range(len(x)):
                out = self.f_forward(x[i], w1, w2)
                l.append((loss(out, Y[i])))
                w1, w2 = self.back_prop(x[i], y[i], w1, w2, alpha)
            print("epochs:", j + 1, "======== acc:", (1 - (sum(l) / len(x))) * 100)
            acc.append((1 - (sum(l) / len(x))) * 100)
            losss.append(sum(l) / len(x))
        return (acc, losss, w1, w2)


if __name__ == '__main__':
    init_params = np.random.random(n_qubit * 4, requires_grad=True)
    opt = qml.AdagradOptimizer(stepsize=0.3)
    params = init_params
    eta_val = np.random.random(1, requires_grad=True)

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
