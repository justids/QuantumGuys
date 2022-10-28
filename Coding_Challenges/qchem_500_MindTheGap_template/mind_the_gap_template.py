import sys
import pennylane as qml
from pennylane import numpy as np
from pennylane import hf


def ground_state_VQE(H):
    """Perform VQE to find the ground state of the H2 Hamiltonian.

    Args:
        - H (qml.Hamiltonian): The Hydrogen (H2) Hamiltonian

    Returns:
        - (float): The ground state energy
        - (np.ndarray): The ground state calculated through your optimization routine
    """

    # QHACK #
    dev = qml.device("default.qubit", wires=[0,1,2,3])
    def circuit(para, wires):
        qml.BasisState(np.array([1,1,0,0]), wires=wires)
        qml.DoubleExcitation(para, wires=wires)
        
    @qml.qnode(dev)
    def cost(para):
        circuit(para, wires=[0,1,2,3])
        return qml.expval(H)
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    th = np.array(0.0, requires_grad=True)
    en = [cost(th)]; ang = [th]
    
    for n in range(100):
        th, prev_en = opt.step_and_cost(cost, th)
        en.append(cost(th)); ang.append(th)
        if abs(en[-1] - prev_en) < 1e-6:
            break
            
    st = np.zeros(16, dtype=np.complex128)
    st[12] = np.cos(th/2); st[3] = -np.sin(th/2)
    return prev_en, st
    # QHACK #


def create_H1(ground_state, beta, H):
    """Create the H1 matrix, then use `qml.Hermitian(matrix)` to return an observable-form of H1.

    Args:
        - ground_state (np.ndarray): from the ground state VQE calculation
        - beta (float): the prefactor for the ground state projector term
        - H (qml.Hamiltonian): the result of hf.generate_hamiltonian(mol)()

    Returns:
        - (qml.Observable): The result of qml.Hermitian(H1_matrix)
    """

    # QHACK #
    mat = np.zeros((16,16), dtype=np.complex128)
    for coef, op in zip(H.coeffs, H.ops):
        mat += coef*qml.utils.expand(op.matrix, op.wires, 4)
    ret = mat + beta*np.outer(ground_state, ground_state)
    return qml.Hermitian(ret,wires=[0,1,2,3])
    # QHACK #


def excited_state_VQE(H1):
    """Perform VQE using the "excited state" Hamiltonian.

    Args:
        - H1 (qml.Observable): result of create_H1

    Returns:
        - (float): The excited state energy
    """

    # QHACK #
    dev = qml.device("default.qubit", wires=[0,1,2,3])
    def circuit(para, wires):
        qml.BasisState(np.array([0,1,0,1]), wires=wires)
        qml.DoubleExcitation(para, wires=wires)
        
    @qml.qnode(dev)
    def cost(para):
        circuit(para, wires=[0,1,2,3])
        return qml.expval(H1)
    
    opt = qml.GradientDescentOptimizer(stepsize=0.4)
    th = np.array(0.0, requires_grad=True)
    en = [cost(th)]; ang = [th]
    
    for n in range(100):
        th, prev_en = opt.step_and_cost(cost, th)
        en.append(cost(th)); ang.append(th)
        if abs(en[-1] - prev_en) < 1e-6:
            break
            
    return prev_en
    # QHACK #


if __name__ == "__main__":
    coord = float(sys.stdin.read())
    symbols = ["H", "H"]
    geometry = np.array([[0.0, 0.0, -coord], [0.0, 0.0, coord]], requires_grad=False)
    mol = hf.Molecule(symbols, geometry)

    H = hf.generate_hamiltonian(mol)()
    E0, ground_state = ground_state_VQE(H)

    beta = 15.0
    H1 = create_H1(ground_state, beta, H)
    E1 = excited_state_VQE(H1)

    answer = [np.real(E0), E1]
    print(*answer, sep=",")
