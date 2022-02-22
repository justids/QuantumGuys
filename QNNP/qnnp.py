import pennylane as qml
from pennylane import numpy as np
import torch

from AtomLoader import *


class QuantumCicuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits, dev):
        self.n_qubits = n_qubits
        self.dev = dev

    def run(self, parameters, idx):
        etas = parameters[0:self.n_qubits].detach().numpy()
        ksis = parameters[self.n_qubits:]

        atom_data = AtomLoader(np.abs(etas), idx=[idx])
        n_atoms = len(atom_data[idx]['descriptor'])

        descriptors = atom_data[idx]['descriptor']

        def apply_descriptor(descriptor):
            for j, description in enumerate(descriptor):
                qml.Rot(
                    description[1],
                    description[0],
                    0,
                    wires=j
                )

        def entangler(params):
            # params: (L, n_qubits) tensor like
            weight = params.reshape((int(len(params)/self.n_qubits), self.n_qubits))
            qml.BasicEntanglerLayers(weights=weight, wires=range(self.n_qubits))

        @qml.qnode(self.dev, interface='torch')
        def measure_energy(params, discriptor=None):
            apply_descriptor(discriptor)
            entangler(params)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)]

        energy = 0
        for i, descriptor in enumerate(descriptors):
            energy += torch.mean(measure_energy(ksis, descriptor))

        return energy

if __name__ == "__main__":

    n_qubits = 2
    depth = 2
    dev = qml.device('default.qubit', wires=n_qubits, shots=2 ** 13)
    qc = QuantumCicuit(n_qubits=n_qubits, dev=dev)
    parameters = torch.rand(2*n_qubits*depth, requires_grad=True)

    loss_fn = torch.nn.MSELoss()

    opt = torch.optim.Adagrad([parameters], lr=0.01)

    idxlist = [str(x) for x in (np.random.choice(len(qm9), 500) + 1)]

    for i, x in enumerate(idxlist):
        est_energy = qc.run(parameters, idx=x)
        list_eta = parameters[:n_qubits]
        atom_data = AtomLoader(np.array(list_eta.tolist()), idx=[x])
        true_energy = atom_data[x]['ground_energy'][0]
        loss = torch.pow((-atom_data[x]['ground_energy'][0] / 20000 - est_energy) / atom_data[x]['atomic_number'], 2)
        if i % 10 == 0:
            print(i, loss.item())
        opt.zero_grad()
        loss.backward()
        opt.step()

