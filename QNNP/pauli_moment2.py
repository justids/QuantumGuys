import sys
import pennylane as qml
from pennylane import numpy as np
from QMLatomLoader import *
from tqdm import tqdm
import json
import numpy




epochs=2000
batchs=8

n_qubit=3
radius=5



# para=np.array([[0,0.5],
#                [0.1,1],
#                [1,5]])








dev = qml.device("default.qubit", wires=n_qubit)

@qml.qnode(dev)
def atomicoutput(descript,H):
    """The energy expectation value of a Hamiltonian"""
    for i in range(n_qubit):
        qml.Rot(-descript[i,1],descript[i,0],descript[i,1],wires=i)
        
    return qml.expval(H)


def hamiltonian(parameters,descriptor_size):
    coeff=np.zeros((n_qubit,4))+descriptor_size[:,np.newaxis]
    coeff=coeff.flatten()
    coeff=coeff*parameters
    obs=[]
    for i in range(n_qubit):
        obs.append(qml.Identity(i))
        obs.append(qml.PauliX(i))
        obs.append(qml.PauliY(i))
        obs.append(qml.PauliZ(i))
    return qml.Hamiltonian(coeff,obs)

# def losses(param,ground_energy,descriptor,descript_sizes,n_atom):

#     outputs=0
#     for n in range(n_atom):     
#         outputs+=atomicoutput(descriptor[n],hamiltonian(param,descript_sizes[n]))
#     return ((ground_energy-outputs)/n_atom)**2

init_params=np.random.random(n_qubit*6,requires_grad=True)
# opt = qml.AdagradOptimizer(stepsize=2)
opt = qml.AdamOptimizer(stepsize=0.3)
params=init_params
losslist=[]


for i in tqdm(range(epochs)):
    def losses(param):
        des_para=param[:2*n_qubit].reshape((n_qubit,2))
        ham_para=param[2*n_qubit:]
        x=AtomLoader2(
        sampler='random',
        epochs=1,
        batchs=batchs,
        classic=True,
        classic_parameter=des_para,
        weigthed=True,
        cutoff_radius=radius
        )
        loss=0
        for j in range(batchs):
            ground_energy=x[j]['ground_energy']
            descriptor=x[j]['descriptor']
            descript_sizes=x[j]['descriptor_size']
            n_atom=x[j]['atomic_number']

            out=0
            for n in range(n_atom):     
                out+=atomicoutput(descriptor[n],hamiltonian(ham_para,descript_sizes[n]))
            loss+=np.sqrt(((ground_energy-out)/n_atom)**2)
        return loss/batchs
    if i%10==0:
    
        params,loss=opt.step_and_cost(losses,params)
        print(loss)
        losslist.append(loss)

    
    else:
        params=opt.step(losses,params)        
print(losslist)
print(params)
paradict={}
paradict["hamiltoninan_para"]=[x for i,x in enumerate(params)]
paradict["loss"]=losslist
with open('paradict.json','w') as fp:
    json.dump(paradict,fp)