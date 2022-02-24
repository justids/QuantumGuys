#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def deutsch_jozsa(fs):
    """Function that determines whether four given functions are all of the same type or not.

    Args:
        - fs (list(function)): A list of 4 quantum functions. Each of them will accept a 'wires' parameter.
        The first two wires refer to the input and the third to the output of the function.

    Returns:
        - (str) : "4 same" or "2 and 2"
    """

    # QHACK #

    dev = qml.device('default.qubit', wires=6, shots=1)

    def prep(index, wires):
        if index == 0:
            qml.PauliX(wires=wires[0])
            qml.PauliX(wires=wires[1])
        elif index == 1:
            qml.PauliX(wires=wires[0])
        elif index == 2:
            qml.PauliX(wires=wires[1])

    @qml.qnode(dev)
    def orig_version_DJ(index):
        qml.PauliX(wires=5)
        [qml.Hadamard(wires=w) for w in [0, 1, 5]]

        def temp():
            fs[index](wires=[0, 1, 5])

        temp()

        [qml.Hadamard(wires=w) for w in [0, 1]]
        return qml.sample(wires=range(2))

    def multioracle():
        qml.PauliX(wires=4)
        [qml.Hadamard(wires=w) for w in range(2, 5)]

        for index in range(len(fs)):
            prep(index, wires=list(range(2)))

            def temp():
                fs[index](wires=range(2, 5))

            qml.ctrl(temp, control=range(2))()
            prep(index, wires=list(range(2)))

        [qml.Hadamard(wires=w) for w in range(2, 4)]
        prep(0, wires=list(range(2, 4)))
        qml.Toffoli(wires=[2, 3, 5])
        prep(0, wires=list(range(2, 4)))

    @qml.qnode(dev)
    def ctrl_version_DJ(index):
        qml.BasisStatePreparation(basis_state=list(map(int, np.binary_repr(index, width=2))), wires=range(2))
        multioracle()
        return qml.sample(wires=range(2))

    @qml.qnode(dev)
    def DJcircuit():
        target_wires = [0, 1, 5]
        qml.PauliX(wires=5)
        [qml.Hadamard(wires=w) for w in target_wires]

        multioracle()

        [qml.Hadamard(wires=w) for w in range(2)]
        return qml.sample(wires=range(2))

    def checker(sample):
        for s in sample:
            if s == 1:
                return 'balanced'
        return 'constant'

    return [(checker(ctrl_version_DJ(index)), checker(orig_version_DJ(index))) for index in range(len(fs))]
    # return checker(DJcircuit())
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    # Definition of the four oracles we will work with.

    def f1(wires):
        qml.CNOT(wires=[wires[numbers[0]], wires[2]])
        qml.CNOT(wires=[wires[numbers[1]], wires[2]])

    def f2(wires):
        qml.CNOT(wires=[wires[numbers[2]], wires[2]])
        qml.CNOT(wires=[wires[numbers[3]], wires[2]])

    def f3(wires):
        qml.CNOT(wires=[wires[numbers[4]], wires[2]])
        qml.CNOT(wires=[wires[numbers[5]], wires[2]])
        qml.PauliX(wires=wires[2])

    def f4(wires):
        qml.CNOT(wires=[wires[numbers[6]], wires[2]])
        qml.CNOT(wires=[wires[numbers[7]], wires[2]])
        qml.PauliX(wires=wires[2])

    output = deutsch_jozsa([f1, f2, f3, f4])
    print(f"{output}")
