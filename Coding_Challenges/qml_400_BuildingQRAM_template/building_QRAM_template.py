#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def qRAM(thetas):
    """Function that generates the superposition state explained above given the thetas angles.

    Args:
        - thetas (list(float)): list of angles to apply in the rotations.

    Returns:
        - (list(complex)): final state.
    """

    # QHACK #

    # Use this space to create auxiliary functions if you need it.

    # QHACK #
    def flipflop(j):
            binaryi=np.array([int(i) for i in list(bin(j))[2:]], requires_grad=False)
            binaryn=np.zeros(3)
            for i in range(len(binaryi)):
                binaryn[-(i+1)]=binaryi[-(i+1)]
            for i in range(3):
                if binaryn[i]==0:
                    qml.PauliX(wires=i)
            U=np.array([[np.cos(thetas[j]/2),-np.sin(thetas[j]/2)]
                        ,[np.sin(thetas[j]/2), np.cos(thetas[j]/2)]])
            qml.ControlledQubitUnitary(U, control_wires=[0, 1, 2], wires=3)
            for i in range(3):
                if binaryn[i]==0:
                    qml.PauliX(wires=i)
            
                    

    dev = qml.device("default.qubit", wires=range(4))

    @qml.qnode(dev)
    def circuit():
        for i in range(3):
            qml.Hadamard(wires=i)
        for j in range(len(thetas)):
            flipflop(j)
            

        # QHACK #

        # Create your circuit: the first three qubits will refer to the index, the fourth to the RY rotation.

        # QHACK #

        return qml.state()

    return circuit()


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    thetas = np.array(inputs, dtype=float)

    output = qRAM(thetas)
    output = [float(i.real.round(6)) for i in output]
    print(*output, sep=",")
