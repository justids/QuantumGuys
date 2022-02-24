import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.autograd import Function
from torchvision import datasets, transforms
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import qiskit
from qiskit import transpile, assemble
from qiskit.visualization import *
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister, transpile, assemble
from qiskit import Aer
from qiskit.tools.visualization import circuit_drawer
from qiskit.quantum_info import state_fidelity
from qiskit import BasicAer
from qiskit.circuit.library import TwoLocal
from AtomLoader import *
from copy import deepcopy


class QuantumCircuit:
    """ 
    This class provides a simple interface for interaction 
    with the quantum circuit 
    """

    def __init__(self, n_qubits, depth):

        self._circuit = qiskit.QuantumCircuit(n_qubits)
        self.n_qubits = n_qubits
        self.depth = depth

        self.backend = Aer.get_backend('aer_simulator')

    def run(self, parameters, idx):

        list_eta = np.array(parameters[0:self.n_qubits])
        list_ksi = np.array(parameters[self.n_qubits:])

        atom_data = AtomLoader(np.abs(list_eta), idx=[idx])

        descriptors = atom_data[idx]['descriptor']
        n_atoms = len(atom_data[idx]['descriptor'])

        # ksi = qiskit.circuit.Parameter('ksi')
        twolocal = TwoLocal(num_qubits=self.n_qubits, reps=self.depth, rotation_blocks=['ry', 'rz'],
                            entanglement_blocks='cx', entanglement='circular', parameter_prefix='ξ',
                            insert_barriers=True)
        twolocal = twolocal.bind_parameters(list_ksi)
        self._circuit.barrier()
        self._circuit = self._circuit.compose(twolocal)
        self._circuit.barrier()
        self._circuit.z(0)
        self._circuit.barrier()
        self._circuit = self._circuit.compose(twolocal.inverse())
        self._circuit.barrier()

        energy = 0
        for i, descriptor in enumerate(descriptors):
            qc_descriptor = qiskit.QuantumCircuit(self.n_qubits)
            for j, description in enumerate(descriptor):
                qc_descriptor.u(
                    description[0],
                    description[1],
                    0,
                    j
                )
            qc = qc_descriptor.compose(self._circuit)
            qc.save_statevector()
            t_qc = transpile(qc,
                             self.backend)
            qobj = assemble(t_qc)
            # parameter_binds = [ksi for ksi in list_ksi])
            job = self.backend.run(qobj)
            result = job.result()

            outputstate = result.get_statevector(qc, decimals=10)
            o = outputstate

            energy += np.real(o[0])

        return energy


class HybridFunction(Function):
    """ Hybrid quantum - classical function definition """

    @staticmethod
    def forward(ctx, input, quantum_circuit, shift, idx):
        """ Forward pass computation """
        ctx.shift = shift
        ctx.quantum_circuit = quantum_circuit

        expectation_z = ctx.quantum_circuit.run(input, idx)
        result = torch.tensor([expectation_z])
        ctx.save_for_backward(input, result)
        ctx.idx = idx

        return result

    @staticmethod
    def backward(ctx, grad_output):
        """ Backward pass computation """
        input, expectation_z = ctx.saved_tensors
        input_list = np.array(input.tolist())
        idx = ctx.idx
        shift_right = input_list + np.ones(input_list.shape) * ctx.shift
        shift_left = input_list - np.ones(input_list.shape) * ctx.shift

        gradients = []
        for i in range(len(input_list)):
            inputlist1 = input_list
            inputlist1[i] = shift_right[i]
            inputlist2 = input_list
            inputlist2[i] = shift_left[i]
            expectation_right = ctx.quantum_circuit.run(inputlist1, idx)
            expectation_left = ctx.quantum_circuit.run(inputlist2, idx)

            gradient = torch.tensor([expectation_right]) - torch.tensor([expectation_left])
            gradients.append(gradient[0])
            print(i)
        gradients = np.array(gradients).T

        return torch.from_numpy(gradients).float() * grad_output.float(), None, None, None


class Hybrid(nn.Module):
    """ Hybrid quantum - classical layer definition """

    def __init__(self, n_qubits, depth, backend, shift):
        super(Hybrid, self).__init__()
        self.quantum_circuit = QuantumCircuit(n_qubits, depth)
        self.shift = shift

    def forward(self, input, idx):
        return HybridFunction.apply(input, self.quantum_circuit, self.shift, idx)


class Net(nn.Module):
    def __init__(self, n_qubits, depth):
        super(Net, self).__init__()
        self.linear = nn.Linear(1, n_qubits * (2 * depth + 3))
        self.hybrid = Hybrid(n_qubits, depth, backend=Aer.get_backend('aer_simulator'), shift=np.pi / 2)

    def forward(self, idx):
        x = self.linear(torch.Tensor([1]))
        return self.hybrid(x, idx)


if __name__ == '__main__':
    N_qubit = 2
    depth = 1
    model = Net(N_qubit, depth=depth)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()

    idxlist = [str(x) for x in (np.random.choice(len(qm9), 500) + 1)]

    for i, x in enumerate(idxlist):
        output = model(x)
        list_eta = model.linear(torch.Tensor([1]))[:N_qubit]
        atom_data = AtomLoader(np.array(list_eta.tolist()), idx=[x])
        optimizer.zero_grad()
        loss = torch.pow((-atom_data[x]['ground_energy'][0] / 20000 - output) / atom_data[x]['atomic_number'], 2)
        loss.backward()
        print(loss)
        optimizer.step()
        print(loss)
