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

    dev = qml.device('default.qubit', wires=8, shots=1)
    def ctrl_state(index, wires):
        assert index<=3
        if index==0:
            pass
        elif index==1:
            qml.PauliX(wires=wires[1])
        elif index==2:
            qml.PauliX(wires=wires[0])
        else:
            qml.PauliX(wires=wires[0])
            qml.PauliX(wires=wires[1])

    wires = list(range(3))
    control_wires = list(range(3, 5))
    ansatz_wires = list(range(5, 8))

    def multioracle(wires, index=None):
        qml.PauliX(wires=ansatz_wires[2])
        [qml.Hadamard(wires=w) for w in ansatz_wires]

        qml.CNOT(wires=[wires[0], control_wires[0]])
        qml.CNOT(wires=[wires[1], control_wires[1]])
        
        if index is None:
            for i, fn in enumerate(fs):
                ctrl_state(i, control_wires)
                def tempf():
                    fn(ansatz_wires)
                qml.ctrl(tempf, control=control_wires)()
                ctrl_state(i, control_wires)
        else:
            ctrl_state(index, control_wires)
            def tempf():
                fs[index](ansatz_wires)
            qml.ctrl(tempf, control=control_wires)()
            ctrl_state(index, control_wires)

        [qml.Hadamard(wires=w) for w in ansatz_wires[:2]]
        qml.MultiControlledX(control_wires=ansatz_wires[:2], wires=wires[2], control_values='00')

    @qml.qnode(dev)
    def test_circuit(index):
        qml.PauliX(wires=wires[2])
        [qml.Hadamard(wires=w) for w in wires]
        multioracle(wires, index)
        [qml.Hadamard(wires=w) for w in wires[:2]]
        return qml.sample(wires=wires[:2])

    def checker(sample):
        for s in sample:
            if s==1:
                return 'balanced'
        return 'constant'

    @qml.qnode(dev)
    def circuit(index):
        qml.PauliX(wires=wires[2])
        [qml.Hadamard(wires=w) for w in wires]
        fs[index](wires)
        [qml.Hadamard(wires=w) for w in wires[:2]]
        return qml.sample(wires=wires[:2])


    return [(checker(test_circuit(index)), checker(circuit(index))) for index in range(4)], 
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
