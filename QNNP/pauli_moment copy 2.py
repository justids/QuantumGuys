import sys
import pennylane as qml
from pennylane import numpy as np
from QMLatomLoader import *
from tqdm import tqdm
import json
import numpy




epochs=1000
batchs=32

n_qubit=3
radius=10



des_para0=np.array(
    [[radius, 0],
     [radius, 1],
     [radius,2]
]
)
des_para0.requires_grad=False
ham_para0=np.array([-0.18460532, -4.87518843, -2.76178594, -0.48570543,0.1071005,  -0.00440127,  0.56326938,  0.08520625])
ham_para0.requires_grad=False




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

init_params=np.random.random(4,requires_grad=True)
# opt = qml.AdagradOptimizer(stepsize=2)
opt = qml.AdamOptimizer(stepsize=0.5)
params=init_params
losslist=[]



for i in tqdm(range(epochs)):
    def losses(param):
        des_para=des_para0
        ham_para=np.concatenate((ham_para0,param),axis=0)
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