#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


def switch(oracle):
    """Function that, given an oracle, returns a list of switches that work by executing a
    single circuit with a single shot. The code you write for this challenge should be completely
    contained within this function between the # QHACK # comment markers.

    Args:
        - oracle (function): oracle that simulates the behavior of the lights.

    Returns:
        - (list(int)): List with the switches that work. Example: [0,2].
    """

    dev = qml.device("default.qubit", wires=[0, 1, 2, "light"], shots=1)

    @qml.qnode(dev)
    def circuit():

        # QHACK #
        qml.Hadamard(wires=[0])
        qml.Hadamard(wires=[1])
        qml.Hadamard(wires=[2])
        qml.PauliX(wires=["light"])
        qml.Hadamard(wires=["light"])
        # You are allowed to place operations before and after the oracle without any problem.
        oracle()
        qml.Hadamard(wires=[0])
        qml.Hadamard(wires=[1])
        qml.Hadamard(wires=[2])
        # QHACK #

        return qml.sample(wires=range(3))

    sample = circuit()

    # QHACK #
    ret = []
    for i in range(3):
        if sample[i] == 1:
            ret.append(i)
    # Process the received sample and return the requested list.
    return ret
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]

    def oracle():
        for i in numbers:
            qml.CNOT(wires=[i, "light"])

    output = switch(oracle)
    print(*output, sep=",")
