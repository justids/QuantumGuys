import sys
import pennylane as qml
from pennylane import numpy as np
from QMLatomLoader import *
from tqdm import tqdm






epochs=1000
batchs=20


para=np.array([[0,0.5],
               [0.1,1],
               [1,5]])

n_qubit=para.shape[0]






dev = qml.device("default.qubit", wires=para.shape[0])

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

init_params=np.random.random(n_qubit*4,requires_grad=True)
opt = qml.AdagradOptimizer(stepsize=0.3)
params=init_params


losslist=[]
if batchs==1:
    loadatom=AtomLoader1(
        sampler='random',
        numb=epochs,
        classic=True,
        classic_parameter=para,
        weigthed=True
        )
    for i in tqdm(range(epochs)):
        ground_energy=loadatom[i]['ground_energy']
        descriptor=loadatom[i]['descriptor']
        descript_sizes=loadatom[i]['descriptor_size']
        n_atom=loadatom[i]['atomic_number']
        descriptor.requires_grad=False
        descript_sizes.requires_grad=False
        def losses(param):    
            outputs=0
            for n in range(n_atom):     
                outputs+=atomicoutput(descriptor[n],hamiltonian(param,descript_sizes[n]))
            return np.sqrt(((ground_energy-outputs)/n_atom)**2)
        
            
    
        
        if i%10==0:
            # print(opt.step_and_cost(losses,params,ground_energy,descriptor,descript_sizes,n_atom))
            params,loss=opt.step_and_cost(losses,params)
            print(loss)
        
        else:
            params=opt.step(losses,params)
else:
    loadatom=AtomLoader2(
        sampler='random',
        epochs=epochs,
        batchs=batchs,
        classic=True,
        classic_parameter=para,
        weigthed=True
        )
    for i in tqdm(range(epochs)):
        def losses(param):
            loss=0
            x=loadatom[i]
            for j in range(batchs):
                ground_energy=x[j]['ground_energy']
                descriptor=x[j]['descriptor']
                descript_sizes=x[j]['descriptor_size']
                n_atom=x[j]['atomic_number']
                descriptor.requires_grad=False
                descript_sizes.requires_grad=False
                out=0
                for n in range(n_atom):     
                    out+=atomicoutput(descriptor[n],hamiltonian(param,descript_sizes[n]))
                loss+=np.sqrt(((ground_energy-out)/n_atom)**2)
            return loss/batchs
        
        if i%10==0:
        
            params,loss=opt.step_and_cost(losses,params)
            print(loss)
            losslist.append(loss)
            p=np.array(losslist)
            print(p)
        
        else:
            params=opt.step(losses,params)        
