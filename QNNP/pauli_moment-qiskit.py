import sys
import numpy as np
from QMLatomLoader_torch import *
from tqdm import tqdm

from qiskit import QuantumCircuit, Aer
from qiskit.utils import QuantumInstance
from qiskit.opflow.primitive_ops import CircuitOp
from qiskit.opflow import (StateFn, Zero, One, Plus, Minus, H, I, X, Y, Z,
                           DictStateFn, VectorStateFn, CircuitStateFn, OperatorStateFn)
from qiskit.opflow import PauliExpectation, CircuitSampler, StateFn

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

epochs = 5
batchs = 5

para = torch.tensor([[7.44546256e+00, 3.58156079e-04],
                     [1.61157272e+01, 7.72828926e-01],
                     [5.79686532e+00, 8.85346477e+00]])

n_qubit = para.shape[0]

backend = Aer.get_backend('qasm_simulator')


def atomicoutput(descript, H):
    """The energy expectation value of a Hamiltonian"""
    psi = QuantumCircuit(n_qubit)
    for i in range(n_qubit):
        psi.rz(-descript[i, 1].item(), i)
        psi.ry(descript[i, 0].item(), i)
        psi.rz(descript[i, 1].item(), i)
    #        qml.Rot(-descript[i,1],descript[i,0],descript[i,1],wires=i)
    # Ref for calculating exp value in qiskit:
    # https://quantumcomputing.stackexchange.com/questions/12080/evaluating-expectation-values-of-operators-in-qiskit

    psi = CircuitStateFn(psi)
    return psi.adjoint().compose(H).compose(psi).eval().real


'''
    q_instance = QuantumInstance(backend, shots=1024)
    measurable_expression = StateFn(H, is_measurement=True).compose(psi) 
    expectation = PauliExpectation().convert(measurable_expression)  
    sampler = CircuitSampler(q_instance).convert(expectation) 
    
    return sampler.eval().real
'''


# idx: i
# typ = 0(I), 1(X), 2(Y), 3(Z)
def obs_term(n_qubit, idx, typ):
    op = I
    if typ == 1:
        op = X
    elif typ == 2:
        op = Y
    elif typ == 3:
        op = Z

    if idx != 0:
        ret = I
    else:
        ret = op

    for i in range(idx - 1):
        ret = ret ^ I

    if idx != 0:
        ret = ret ^ op

    for i in range(n_qubit - idx - 1):
        ret = ret ^ I

    return ret


def hamiltonian(parameters, descriptor_size):
    #    print(descriptor_size[:,np.newaxis])
    #    coeff=np.zeros((n_qubit,4))+np.array(descriptor_size[:,np.newaxis])
    coeff = np.zeros((n_qubit, 4)) * 10000
    coeff = coeff.flatten()
    for i in range(n_qubit * 4):
        coeff[i] = coeff[i] * (parameters[i].item())
    obs = 0
    for i in range(n_qubit):
        if i == 0:
            obs = coeff[0] * obs_term(n_qubit, i, 0)  # I
        else:
            obs = (obs) + (coeff[4 * i] * obs_term(n_qubit, i, 0))  # I
        obs = (obs) + (coeff[4 * i + 1] * obs_term(n_qubit, i, 1))  # X
        obs = (obs) + (coeff[4 * i + 2] * obs_term(n_qubit, i, 2))  # Y
        obs = (obs) + (coeff[4 * i + 3] * obs_term(n_qubit, i, 3))  # Z
    return obs

    # def losses(param,ground_energy,descriptor,descript_sizes,n_atom):

    #     outputs=0
    #     for n in range(n_atom):
    #         outputs+=atomicoutput(descriptor[n],hamiltonian(param,descript_sizes[n]))
    #     return ((ground_energy-outputs)/n_atom)**2


class DesNet(nn.Module):
    def __init__(self):
        super(DesNet, self).__init__()
        self.r_function1 = nn.Linear(1, 5)
        self.r_function2 = nn.Linear(5, 1)

        self.z_function1 = nn.Linear(1, 5)
        self.z_function2 = nn.Linear(5, 1)

    def forward(self, Rij: torch.Tensor, Zj: torch.Tensor):
        """

            Args:
                Rij: N_atom x N_atom distance matrix
                Zj: N_atom atom number matrix

            Returns:
                N_atom descriptor
            """
        n_atoms = len(Zj)
        Rij = Rij.view(n_atoms * n_atoms, 1)
        for i in range(n_atoms * n_atoms):
            _hidden = F.relu(self.r_function1(Rij[i]))
            Rij[i] = F.relu(self.r_function2(_hidden))

        Rij = Rij.reshape(n_atoms, n_atoms)

        Zj = Zj.view(n_atoms, 1)
        for i in range(n_atoms):
            _Z = F.relu(self.z_function1(Zj[i]))
            Zj[i] = F.relu(self.z_function2(_Z))
        Zj = Zj.reshape(n_atoms)

        ret = torch.matmul(Zj, Rij)  # descriptor (N_atom,)
        return ret


if __name__ == '__main__':
    init_params = torch.rand(n_qubit * 4, requires_grad=True)

    # opt = qml.AdagradOptimizer(stepsize=0.3)
    # init_params = torch.tensor(
    #     [-0.47594351, -4.50743477, -8.46613537, -0.19065087, -1.1726193, -11.25678421, 14.01949377, -8.60061223,
    #      -0.25404365, 0.23194815, 0.10643894, 0.74158512], requires_grad=True)
    params = init_params
    desnet = DesNet()
    desnet.train()
    net_params = desnet.parameters()
    opt = optim.Adagrad([params], lr=0.01)

    losslist = []
    if batchs == 1:
        loadatom = AtomLoader1(
            sampler='random',
            numb=epochs,
            # classic=True,
            classic=False,
            # classic_parameter=para,
            # weigthed=True
            desnet=desnet
        )
        for i in tqdm(range(epochs)):
            ground_energy = loadatom[i]['ground_energy']
            descriptor = loadatom[i]['descriptor']
            descript_sizes = loadatom[i]['descriptor_size']
            n_atom = loadatom[i]['atomic_number']
            descriptor.requires_grad = False
            descript_sizes.requires_grad = False


            def losses(param):
                outputs = 0
                for n in range(n_atom):
                    outputs += atomicoutput(descriptor[n], hamiltonian(param, descript_sizes[n]))
                return torch.tensor(np.sqrt(((ground_energy - outputs) / n_atom) ** 2))


            def closure(pt=False):
                opt.zero_grad()
                loss = losses(params)
                # loss.requires_grad_(True)
                loss.backward()
                return loss


            opt.step(closure)
            if i % 10 == 0:
                loss = losses(params).item()
                losslist.append(loss)
                print(losslist)

    else:
        loadatom = AtomLoader2(
            sampler='random',
            epochs=epochs,
            batchs=batchs,
            # classic=True,
            classic=False,
            # classic_parameter=para,
            # weigthed=True
            desnet=desnet
        )
        for i in tqdm(range(epochs)):
            def losses(param):
                loss = 0
                x = loadatom[i]
                for j in range(batchs):
                    ground_energy = x[j]['ground_energy']
                    descriptor = x[j]['descriptor']
                    descript_sizes = x[j]['descriptor_size']
                    n_atom = x[j]['atomic_number']
                    # descriptor.requires_grad = False
                    # descript_sizes.requires_grad = False
                    out = 0
                    for n in range(n_atom):
                        out += atomicoutput(descriptor[n], hamiltonian(param, descript_sizes[n]))
                    loss += np.sqrt(((ground_energy - out) / n_atom) ** 2)
                return torch.abs(torch.tensor(loss / batchs))


            def closure():
                opt.zero_grad()
                loss = losses(params)
                # loss.requires_grad_(True)
                loss.backward()
                return loss


            opt.step(closure)
            if 0 == 0:
                loss = losses(params).item()
                losslist.append(loss)
                print(loss)
