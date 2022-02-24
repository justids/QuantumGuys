import sys
import pennylane as qml
from pennylane import numpy as np
from QMLatomLoader import *
from tqdm import tqdm
import json





epochs=1500
batchs=32

n_qubit=3
radius=5


para=np.array([[7.44546256e+00, 3.58156079e-04],
        [1.61157272e+01, 7.72828926e-01],
        [5.79686532e+00, 8.85346477e+00]], requires_grad=False)


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

init_params=np.random.random(n_qubit*4,requires_grad=True)
opt = qml.AdagradOptimizer(stepsize=1)
params=[
    -0.10427488,  0.39103157,  2.00803408, -1.03417637, -0.3711771,   0.11335513,
    1.31828257,  3.19598659, -0.25595048, -1.06790523,  0.05467634, -0.09911154,
    0.63642456, -1.25679513,  1.39696794,  1.14624053, -0.02484633,  0.15336306,
    0.22153725, -0.09010125, -0.16219164, -1.54976815,  0.03012084, -0.03090227]


losslist=[]
if batchs==1:
    loadatom=AtomLoader1(
        sampler='random',
        numb=epochs,
        classic=True,
        classic_parameter=para,
        weigthed=True,
        cutoff_radius=radius
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
        weigthed=True,
        cutoff_radius=radius
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

        
        else:
            params=opt.step(losses,params)        
losslist=np.array(losslist)
print(para)
print(losslist)
print(params)
paradict={}
paradict["descriptor_para"]=para
paradict["hamiltoninan_para"]=params
paradict["loss"]=losslist
with open('paradict3.json','w') as fp:
    json.dump(paradict,fp)