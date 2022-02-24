import sys
import pennylane as qml
from pennylane import numpy as np
from QMLatomLoader import *
from tqdm import tqdm
import json

epochs = 2000
batchs = 32

n_qubit = 1
radius = 10

init_desparams = np.abs(np.random.random((n_qubit, 2), requires_grad=True))
desopt = qml.AdagradOptimizer(stepsize=1)
desparams = init_desparams
batch = 64


def descost(paras):
    parass = np.abs(paras)
    return paraoptim(batchs=batch, classic_parameter=parass, weigthed=True, cutoff_radius=radius)


# for i in tqdm(range(300)):
#     desparams=desopt.step(descost,desparams)
#     desparams=np.abs(desparams)
# desparams,descostout=desopt.step_and_cost(descost,desparams)
# desparams=np.abs(desparams)
para = np.array(
    [[radius, 0],
     ]
)
para.requires_grad = False
print(para)

# para=np.array([[0,0.5],
#                [0.1,1],
#                [1,5]])


dev = qml.device("default.qubit", wires=n_qubit)


@qml.qnode(dev)
def atomicoutput(descript, H):
    """The energy expectation value of a Hamiltonian"""
    for i in range(n_qubit):
        qml.Rot(-descript[i, 1], descript[i, 0], descript[i, 1], wires=i)

    return qml.expval(H)


def hamiltonian(parameters, descriptor_size):
    coeff = np.zeros((n_qubit, 4)) + descriptor_size[:, np.newaxis]
    coeff = coeff.flatten()
    coeff = coeff * parameters
    obs = []
    for i in range(n_qubit):
        obs.append(qml.Identity(i))
        obs.append(qml.PauliX(i))
        obs.append(qml.PauliY(i))
        obs.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeff, obs)


# def losses(param,ground_energy,descriptor,descript_sizes,n_atom):

#     outputs=0
#     for n in range(n_atom):     
#         outputs+=atomicoutput(descriptor[n],hamiltonian(param,descript_sizes[n]))
#     return ((ground_energy-outputs)/n_atom)**2

init_params = np.random.random(n_qubit * 4, requires_grad=True)
opt = qml.AdagradOptimizer(stepsize=2)
params = init_params
params.requires_grad = True
print(params)
losslist = []
if batchs == 1:
    loadatom = AtomLoader1(
        sampler='random',
        numb=epochs,
        classic=True,
        classic_parameter=para,
        weigthed=True,
        cutoff_radius=radius
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
            return np.sqrt(((ground_energy - outputs) / n_atom) ** 2)


        if i % 10 == 0:
            # print(opt.step_and_cost(losses,params,ground_energy,descriptor,descript_sizes,n_atom))
            params, loss = opt.step_and_cost(losses, params)
            print(loss)

        else:
            params = opt.step(losses, params)
else:
    loadatom = AtomLoader2(
        sampler='random',
        epochs=epochs,
        batchs=batchs,
        classic=True,
        classic_parameter=para,
        weigthed=True,
        cutoff_radius=radius
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
                descriptor.requires_grad = False
                descript_sizes.requires_grad = False
                out = 0
                for n in range(n_atom):
                    out += atomicoutput(descriptor[n], hamiltonian(param, descript_sizes[n]))
                loss += np.sqrt(((ground_energy - out) / n_atom) ** 2)
            return loss / batchs


        if i % 10 == 0:

            params, loss = opt.step_and_cost(losses, params)
            print(loss)
            losslist.append(loss)


        else:
            params = opt.step(losses, params)
losslist = np.array(losslist)
print(para)
print(losslist)
print(params)
paradict = {}
paradict["descriptor_para"] = para
paradict["hamiltoninan_para"] = params
paradict["loss"] = losslist
with open('paradict.json', 'w') as fp:
    json.dump(paradict, fp)
