import os
import numpy as np
import json
import math
from PyAstronomy import pyasl

an = pyasl.AtomicNo()


with open('output.json','r') as f:
    qm9=json.load(f)

epsilon=1e-15

def rotation_matrix(raxis='z',theta=0):
    dim=theta.shape[0]
    rmatrix= np.zeros((dim, dim, 3, 3))
    if raxis=='x':
        rmatrix[:,:,0,0]=1
        rmatrix[:,:,1,1]=np.cos(theta)
        rmatrix[:,:,1,2]=-np.sin(theta)
        rmatrix[:,:,2,1]=np.sin(theta)
        rmatrix[:,:,2,2]=np.cos(theta)
        return rmatrix
    if raxis=='y':
        rmatrix[:,:,0,0]=np.cos(theta)
        rmatrix[:,:,0,2]=np.sin(theta)
        rmatrix[:,:,1,1]=1
        rmatrix[:,:,2,0]=-np.sin(theta)
        rmatrix[:,:,2,2]=np.cos(theta)
        return rmatrix
    if raxis=='z':
        rmatrix[:,:,0,0]=np.cos(theta)
        rmatrix[:,:,0,1]=-np.sin(theta)
        rmatrix[:,:,1,0]=np.sin(theta)
        rmatrix[:,:,1,1]=np.cos(theta)
        rmatrix[:,:,2,2]=1
        return rmatrix

def Set_center(molecule):
    cord=np.array(molecule["coordinates"])[:,1:].astype(float)
    atomic_num=np.zeros(len(cord[:,0]))
    for i in range(len(cord[:,0])):
        atomic_num[i]=an.getAtomicNo(molecule["coordinates"][i][0])
    distance_martix=np.sqrt(np.sum(np.power(cord[np.newaxis,:,:]-cord[:,np.newaxis,:],2),axis=2))
    # (n,n)  
    bloch_cord=(cord[:,np.newaxis,:]-cord[np.newaxis,:,:]+epsilon)/(distance_martix[:,:,np.newaxis]+epsilon)
    # (n,n,3)
    bloch_theta=np.arccos(bloch_cord[:,:,2])
    # (n,n)
    bloch_pi=np.arctan2(bloch_cord[:,:,1],bloch_cord[:,:,0])
    # (n,n)
    min_point=np.argsort(distance_martix,axis=1)[:,:3]

    theta1=np.array([bloch_theta[i,x] for i,x in enumerate(min_point[:,1])])
    theta2=np.array([bloch_theta[i,x] for i,x in enumerate(min_point[:,2])])
    pi1=np.array([bloch_pi[i,x] for i,x in enumerate(min_point[:,1])])
    pi2=np.array([bloch_pi[i,x] for i,x in enumerate(min_point[:,2])])
    pi_prime=np.arctan2(np.sin(theta2)*np.sin(pi2-pi1),np.cos(theta1)*np.sin(theta2)*np.cos(pi2-pi1)-np.sin(theta1)*np.cos(theta2))
    rot3=rotation_matrix('z',-pi_prime[:,np.newaxis])
    rot2=rotation_matrix('y',(-theta1)[:,np.newaxis])
    rot1=rotation_matrix('z',(-pi1)[:,np.newaxis])
    sym_bloch_cord=np.einsum('abij,abjk,abkl,abl-> abi',rot3,rot2,rot1,bloch_cord)
    sym_bloch_cord[(np.abs(sym_bloch_cord)<epsilon)]=0
    for i in range(len(sym_bloch_cord[:,0,0])):
        sym_bloch_cord[i,i,:]=0
    return sym_bloch_cord, distance_martix, atomic_num, molecule['energy_U0']


def Cutoff_function(distance, cutoff_radius=5):
    return 0.5*(np.cos(math.pi*distance/cutoff_radius)+1)


def Cal_descriptor(cord,distance_matrix,etta,weigthed=False, atomic_num=None, cutoff_radius=5, halve=False):
    descriptor=cord*Cutoff_function(distance_matrix[:,:,np.newaxis],cutoff_radius=cutoff_radius)
    descriptor=descriptor[:,np.newaxis,:,:]*np.exp(-etta[np.newaxis,:,np.newaxis,np.newaxis]*np.power(distance_matrix[:,np.newaxis,:,np.newaxis]-cutoff_radius,2))
    if weigthed==True:
        atomic_weight=np.zeros((len(atomic_num),len(atomic_num)))
        for i in range(len(atomic_num)):
            for j in range(len(atomic_num)):
                atomic_weight[i,j]=atomic_num[i]*atomic_num[j]
        descriptor=descriptor*atomic_weight[:,np.newaxis,:,np.newaxis]
    descriptor=np.sum(descriptor,axis=2)
    descriptor=descriptor/np.sqrt(np.sum(np.power(descriptor,2),axis=2))[:,:,np.newaxis]
    descriptor_theta=np.arccos(descriptor[:,:,2])
    if halve==True: 
        descriptor_theta=descriptor_theta/2
    descriptor_phi=np.arctan2(descriptor[:,:,1],descriptor[:,:,0])
    
    return np.stack((descriptor_theta,descriptor_phi),axis=2)



def AtomLoader(etta,sampler=None,idx=None,numb=1,weigthed=False,cutoff_radius=5, halve=False):
    if sampler==None:
        atomloader={}
        for i,x in enumerate(idx):
            sym_bloch_cord, distance_martix, atomic_num, ground_energy=Set_center(qm9[x])
            descriptor=Cal_descriptor(cord=sym_bloch_cord,distance_matrix=distance_martix,etta=etta,weigthed=weigthed,atomic_num=atomic_num,cutoff_radius=cutoff_radius,halve=halve)
            atomloader[x]={
                                'ground_energy' : ground_energy,
                                'descriptor' : descriptor,
                                'atomic_number': qm9[x]['n_atoms']
                                    }
        return atomloader
    if sampler=='random':
        idx=(np.random.randint(133885,size=numb)+1).astype(str)
        atomloader={}
        for i,x in enumerate(idx):
            sym_bloch_cord, distance_martix, atomic_num, ground_energy=Set_center(qm9[x])
            descriptor=Cal_descriptor(cord=sym_bloch_cord,distance_matrix=distance_martix,etta=etta,weigthed=weigthed,atomic_num=atomic_num,cutoff_radius=cutoff_radius,halve=halve)
            atomloader[x]={
                                'ground_energy' : ground_energy,
                                'descriptor' : descriptor,
                                'atomic_number': qm9[x]['n_atoms']
                                    }
        return atomloader
    