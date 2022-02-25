# Predictiong Ground-state with Novel Quantum Descriptor of Molecules using QML
Qhack Open Hackathon (team name: SexyQuantumGuys)

## Packages
Pennylane, Qiskit, PyTorch, etc.

## Tutorial & Presentation
Refer to [QNNP_Tutorial.ipynb](QNNP/QNNP_Tutorial.ipynb)

## Demonstration
Our main project directory is [QNNP](QNNP).
Try running [pauli_moment.py](QNNP/pauli_moment.py) in that directory as,
```bash
python3 ./QNNP/pauli_moment.py
```
you can adjust size of batch, epochs, and number of qubit used by changing global variables;
`batchs`, `epochs`, `n_qubit`.


## Method
PennyLane [pauli_moment.py](QNNP/pauli_moment.py)
Qiskit    [pauli_moment-qiskit.py](QNNP/pauli_moment-qiskit.py)
