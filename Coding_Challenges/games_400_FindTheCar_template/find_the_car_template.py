#! /usr/bin/python3

import sys
from pennylane import numpy as np
import pennylane as qml


dev = qml.device("default.qubit", wires=[0, 1, "sol"], shots=1)


def find_the_car(oracle):
    """Function which, given an oracle, returns which door that the car is behind.

    Args:
        - oracle (function): function that will act as an oracle. The first two qubits (0,1)
        will refer to the door and the third ("sol") to the answer.

    Returns:
        - (int): 0, 1, 2, or 3. The door that the car is behind.
    """

    @qml.qnode(dev)
    def circuit1():
        # QHACK #
<<<<<<< HEAD
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.PauliX(wires=0)
       
        oracle()
        qml.Hadamard(wires=0)
        qml.Hadamard(wires=1)
        qml.PauliX(wires=0)

        # TODO: Yeah, this is about bernstein-vazirani algorithm (https://qiskit.org/textbook/ch-algorithms/bernstein-vazirani.html)
        
=======
        qml.Hadamard(wires=[0])
        oracle()
        qml.Hadamard(wires=[0])
>>>>>>> 14361838fef348b28dfdb279ba93b61ceadc6c96
        # QHACK #
        return qml.sample()

    @qml.qnode(dev)
    def circuit2():
        # QHACK #
<<<<<<< HEAD
        
        
        oracle()
        qml.RX(np.pi/4,wires = 0)
        

=======
        qml.Hadamard(wires=[1])
        oracle()
        qml.Hadamard(wires=[1])
>>>>>>> 14361838fef348b28dfdb279ba93b61ceadc6c96
        # QHACK #
        return qml.sample()

    sol1 = circuit1()
    sol2 = circuit2()
    print(sol1)
    print(sol2)
    
    # if (sol1==[0,1,0]) and (sol2==[1,1,1]):
    #     return 0

    # QHACK #
    # process sol1 and sol2 to determine which door the car is behind.
    if sol1[0] == 0:
        if sol2[1] == 0:
            return 3
        return 1
    if sol2[1] == 0:
        return 2
    return 0
    # QHACK #


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block
    inputs = sys.stdin.read().split(",")
    numbers = [int(i) for i in inputs]
    
    for numbers in ([0,0],[0,1],[1,0],[1,1]):
        def oracle():
            if numbers[0] == 1:
                qml.PauliX(wires=0)
            if numbers[1] == 1:
                qml.PauliX(wires=1)
            qml.Toffoli(wires=[0, 1, "sol"])
            if numbers[0] == 1:
                qml.PauliX(wires=0)
            if numbers[1] == 1:
                qml.PauliX(wires=1)
        output = find_the_car(oracle)
    # print(f"{output}")
