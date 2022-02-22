import pennylane as qml
from pennylane import numpy as np
import torch

from AtomLoader import *


class QuantumCicuit:
    """
    This class provides a simple interface for interaction
    with the quantum circuit
    """

    def __init__(self, n_qubits):
        self.n_qubits = n_qubits
        self.dev = qml.device('default.qubits', wires=n_qubits, shots=2 ** 13)

    def run(self, parameters, idx):
        etas = parameters[0:self.n_qubits]
        ksis = parameters[self.n_qubits:]

        atom_data = AtomLoader(np.abs(etas), idx=[idx])

        descriptors = atom_data[idx]['descriptor']
        n_atoms = len(atom_data[idx]['descriptor'])

        def apply_descriptor(descriptor):
            for j, description in enumerate(descriptor):
                qml.Rot(
                    description[1],
                    description[0],
                    0,
                    j
                )

        def entangler(params):
            # params: (L, n_qubits) tensor like
            assert len(params.shape) == 2
            qml.BasicEntanglerLayers(weights=params)

        @qml.qnode(self.dev, interface='torch')
        def measure_energy(params, discriptor=None):
            apply_descriptor(discriptor)
            entangler(params)
            return np.mean([qml.expval(qml.PauliZ(i)) for i in range(self.n_qubits)])

        energy = 0
        for i, descriptor in enumerate(descriptors):
            energy += measure_energy(ksis, descriptor)

        return energy


if __name__ == "__main__":

    n_qubits = 2
    depth = 2
    dev = qml.device('default.qubits', wires=n_qubits, shots=2 ** 13)
    qc = QuantumCicuit(n_qubits=n_qubits)
    parameters = torch.rand(depth, n_qubits, requires_grad=True)

    opt = torch.optim.Adam(parameters, lr=0.0001)

    def closure():
        opt.zero_grad()
        loss = qc.run(parameters)

    idxlist = [str(x) for x in (np.random.choice(len(qm9), 500) + 1)]

    for i, x in enumerate(idxlist):
        qc.run()

