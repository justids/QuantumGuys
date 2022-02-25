import os
from pennylane import numpy as np
import json
import math
from PyAstronomy import pyasl
from multiprocessing import Pool, Process, Queue

an = pyasl.AtomicNo()

with open('output.json', 'r') as f:
    qm9 = json.load(f)

epsilon = 1e-15


def rotation_matrix(raxis='z', theta=0):
    dim = theta.shape[0]
    rmatrix = np.zeros((dim, dim, 3, 3))
    if raxis == 'x':
        rmatrix[:, :, 0, 0] = 1
        rmatrix[:, :, 1, 1] = np.cos(theta)
        rmatrix[:, :, 1, 2] = -np.sin(theta)
        rmatrix[:, :, 2, 1] = np.sin(theta)
        rmatrix[:, :, 2, 2] = np.cos(theta)
        return rmatrix
    if raxis == 'y':
        rmatrix[:, :, 0, 0] = np.cos(theta)
        rmatrix[:, :, 0, 2] = np.sin(theta)
        rmatrix[:, :, 1, 1] = 1
        rmatrix[:, :, 2, 0] = -np.sin(theta)
        rmatrix[:, :, 2, 2] = np.cos(theta)
        return rmatrix
    if raxis == 'z':
        rmatrix[:, :, 0, 0] = np.cos(theta)
        rmatrix[:, :, 0, 1] = -np.sin(theta)
        rmatrix[:, :, 1, 0] = np.sin(theta)
        rmatrix[:, :, 1, 1] = np.cos(theta)
        rmatrix[:, :, 2, 2] = 1
        return rmatrix


def Set_center(molecule, set_axis=False):
    cord = np.array(molecule["coordinates"])[:, 1:].astype(float)
    atomic_num = np.zeros(len(cord[:, 0]))
    for i in range(len(cord[:, 0])):  # 원자 개수 n
        atomic_num[i] = an.getAtomicNo(molecule["coordinates"][i][0])
    distance_martix = np.sqrt(np.sum(np.power(cord[np.newaxis, :, :] - cord[:, np.newaxis, :], 2), axis=2))
    # (n,n)  
    bloch_cord = (cord[:, np.newaxis, :] - cord[np.newaxis, :, :] + epsilon) / (
            distance_martix[:, :, np.newaxis] + epsilon)  # n,n,3
    if not set_axis:
        return bloch_cord, distance_martix, atomic_num, molecule['energy_U0']
    # (n,n,3)
    elif set_axis:  # 가장 가까운 원자를 z축 위에, 두번쨰로 가짜운 원자는 xz 평면 위에 올리기(rotation sym 맞추려고 강제)
        bloch_cord[bloch_cord[:, :, 2] > 1, 2] = 1
        bloch_cord[bloch_cord[:, :, 2] < -1, 2] = -1
        bloch_theta = np.arccos(bloch_cord[:, :, 2])
        # (n,n)
        bloch_pi = np.arctan2(bloch_cord[:, :, 1], bloch_cord[:, :, 0])
        # (n,n)
        min_point = np.argsort(distance_martix, axis=1)[:, :3]

        theta1 = np.array([bloch_theta[i, x] for i, x in enumerate(min_point[:, 1])])
        theta2 = np.array([bloch_theta[i, x] for i, x in enumerate(min_point[:, 2])])
        pi1 = np.array([bloch_pi[i, x] for i, x in enumerate(min_point[:, 1])])
        pi2 = np.array([bloch_pi[i, x] for i, x in enumerate(min_point[:, 2])])
        pi_prime = np.arctan2(np.sin(theta2) * np.sin(pi2 - pi1),
                              np.cos(theta1) * np.sin(theta2) * np.cos(pi2 - pi1) - np.sin(theta1) * np.cos(theta2))
        rot3 = rotation_matrix('z', -pi_prime[:, np.newaxis])
        rot2 = rotation_matrix('y', (-theta1)[:, np.newaxis])
        rot1 = rotation_matrix('z', (-pi1)[:, np.newaxis])
        sym_bloch_cord = np.einsum('abij,abjk,abkl,abl-> abi', rot3, rot2, rot1, bloch_cord)
        sym_bloch_cord[(np.abs(sym_bloch_cord) < epsilon)] = 0
        for i in range(len(sym_bloch_cord[:, 0, 0])):
            sym_bloch_cord[i, i, :] = 0
        return sym_bloch_cord, distance_martix, atomic_num, molecule['energy_U0']


def Cutoff_function(distance, cutoff_radius=5):
    return 0.5 * (np.cos(math.pi * distance / cutoff_radius) + 1)


def calqubit(theta, phi):
    return np.stack((np.cos(theta / 2), np.sin(theta / 2) * np.exp(1j * phi)), axis=-1)


def returnangle(qubit):
    the = 2 * np.arctan2(np.abs(qubit[:, :, 1]), np.abs(qubit[:, :, 0]))
    ph = np.arctan2(np.imag(qubit[:, :, 1]), np.real(qubit[:, :, 1]))
    return np.stack((the, ph), axis=2)


def Cal_descriptor(cord, distance_matrix, classic=False, new_parameter=None, classic_parameter=None, weigthed=False,
                   atomic_num=None, cutoff_radius=5, halve=False):
    if classic:
        cord[cord[:, :, 2] > 1, 2] = 1
        cord[cord[:, :, 2] < -1, 2] = -1
        descriptor = calqubit(np.arccos(cord[:, :, 2]), np.arctan2(cord[:, :, 1], cord[:, :, 0])) * Cutoff_function(
            distance_matrix[:, :, np.newaxis], cutoff_radius=cutoff_radius)
        descriptor = descriptor[:, np.newaxis, :, :] * np.exp(
            -classic_parameter[np.newaxis, :, np.newaxis, np.newaxis, 0] * np.power(
                distance_matrix[:, np.newaxis, :, np.newaxis] -
                classic_parameter[np.newaxis, :, np.newaxis, np.newaxis, 1], 2
            )
        )
        if weigthed:
            atomic_weight = np.zeros((len(atomic_num), len(atomic_num)))
            for i in range(len(atomic_num)):
                for j in range(len(atomic_num)):
                    atomic_weight[i, j] = atomic_num[i] * atomic_num[j]
            descriptor = descriptor * atomic_weight[:, np.newaxis, :, np.newaxis]
    else:
        cord[cord[:, :, 2] > 1, 2] = 1
        cord[cord[:, :, 2] < -1, 2] = -1
        descriptor = calqubit(np.arccos(cord[:, :, 2]), np.arctan2(cord[:, :, 1], cord[:, :, 0]))
        descriptor = descriptor[:, np.newaxis, :, :] * new_parameter[np.newaxis, :, np.newaxis, np.newaxis]
    descriptor = np.sum(descriptor, axis=2)
    descript_size = np.sum(descriptor * np.conj(descriptor), axis=2)
    descriptor = (descriptor + epsilon) / (np.sqrt(descript_size[:, :, np.newaxis]) + epsilon)

    return returnangle(descriptor), descript_size


def AtomLoader1(sampler=None, idx=None, numb=1, classic=False, new_parameter=None, classic_parameter=None,
                weigthed=False, cutoff_radius=5, halve=False, set_axis=False):
    if sampler is None:
        atomloader = {}
        for i, x in enumerate(idx):
            sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[x.astype(str)],
                                                                                    set_axis=set_axis)
            descriptor, descript_size = Cal_descriptor(
                cord=sym_bloch_cord,
                distance_matrix=distance_martix,
                classic=classic,
                new_parameter=new_parameter,
                classic_parameter=classic_parameter,
                weigthed=weigthed,
                atomic_num=atomic_num,
                cutoff_radius=cutoff_radius,
                halve=halve
            )
            atomloader[i] = {
                'ground_energy': ground_energy,
                'descriptor': descriptor,
                'descriptor_size': descript_size,
                'atomic_number': qm9[x]['n_atoms']
            }
        return atomloader
    if sampler == 'random':
        idx = (np.random.randint(133885, size=numb) + 1).astype(str)
        atomloader = {}
        for i, x in enumerate(idx):
            sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[str(x)], set_axis=set_axis)
            descriptor, descript_size = Cal_descriptor(

                cord=sym_bloch_cord,
                distance_matrix=distance_martix,
                classic=classic,
                new_parameter=new_parameter,
                classic_parameter=classic_parameter,
                weigthed=weigthed,
                atomic_num=atomic_num,
                cutoff_radius=cutoff_radius,
                halve=halve
            )
            atomloader[i] = {
                'ground_energy': ground_energy,
                'descriptor': descriptor,
                'descriptor_size': descript_size,
                'atomic_number': qm9[str(x)]['n_atoms']
            }
        return atomloader


def AtomLoader2(sampler=None, idx=None, epochs=1, batchs=1, classic=False, new_parameter=None, classic_parameter=None,
                weigthed=False, cutoff_radius=5, halve=False, set_axis=False):
    if sampler is None:
        atomloader = {}
        for i, x in enumerate(idx):
            sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[x.astype(str)],
                                                                                    set_axis=set_axis)
            descriptor, descript_size = Cal_descriptor(
                cord=sym_bloch_cord,
                distance_matrix=distance_martix,
                classic=classic,
                new_parameter=new_parameter,
                classic_parameter=classic_parameter,
                weigthed=weigthed,
                atomic_num=atomic_num,
                cutoff_radius=cutoff_radius,
                halve=halve
            )
            atomloader[i] = {
                'ground_energy': ground_energy,
                'descriptor': descriptor,
                'descriptor_size': descript_size,
                'atomic_number': qm9[x]['n_atoms']
            }
        return atomloader
    if sampler == 'random':
        idx = (np.random.randint(133885, size=epochs * batchs) + 1).astype(str)
        atomloader = {}
        k = 0
        for i in range(epochs):
            atomload = {}
            for j in range(batchs):
                sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[str(idx[k])],
                                                                                        set_axis=set_axis)
                descriptor, descript_size = Cal_descriptor(
                    cord=sym_bloch_cord,
                    distance_matrix=distance_martix,
                    classic=classic,
                    new_parameter=new_parameter,
                    classic_parameter=classic_parameter,
                    weigthed=weigthed,
                    atomic_num=atomic_num,
                    cutoff_radius=cutoff_radius,
                    halve=halve
                )
                atomload[j] = {
                    'ground_energy': ground_energy,
                    'descriptor': descriptor,
                    'descriptor_size': descript_size,
                    'atomic_number': qm9[str(idx[k])]['n_atoms']
                }
                k += 1

            atomloader[i] = atomload
        if epochs == 1:
            return atomload
        else:
            return atomloader


# num_cores = 24

# def paraLoader(classic_parameter=None, weigthed=False,cutoff_radius=5):
#     total=0
#     def caldot(x,q):
#         out=0
#         for i,y in enumerate(array_split[x]):
#             sym_bloch_cord, distance_martix, atomic_num, ground_energy=Set_center(qm9[str(y)])
#             descriptor,descript_size=Cal_descriptor(
#                 cord=sym_bloch_cord,
#                 distance_matrix=distance_martix,
#                 classic=True,
#                 classic_parameter=classic_parameter,
#                 weigthed=weigthed,
#                 atomic_num=atomic_num,
#                 cutoff_radius=cutoff_radius,
#                 )
#             des=np.array([np.sin(descriptor[:,:,0])*np.cos(descriptor[:,:,1]),np.sin(descriptor[:,:,0])*np.sin(descriptor[:,:,1]),np.cos(descriptor[:,:,0])])
#             out+=np.sum(des[:,:,np.newaxis,:]*des[:,np.newaxis,:,:])
#         q.put(out)
#     q=Queue()
#     for i in range(num_cores):
#         proc=Process(target=caldot,args=(i,q))
#         proc.start()
#     for i in range(num_cores):
#         total=total+q.get()

#     return total


def paraoptim(batchs=10, classic_parameter=None, weigthed=False, cutoff_radius=5, set_axis=False):
    idx = (np.random.randint(133885, size=batchs) + 1).astype(str)
    total = 0
    for i, y in enumerate(idx):
        sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[str(y)], set_axis=set_axis)
        descriptor, descript_size = Cal_descriptor(
            cord=sym_bloch_cord,
            distance_matrix=distance_martix,
            classic=True,
            new_parameter=None,
            classic_parameter=classic_parameter,
            weigthed=weigthed,
            atomic_num=atomic_num,
            cutoff_radius=cutoff_radius,
            halve=False
        )
        des = np.stack((np.sin(descriptor[:, :, 0]) * np.cos(descriptor[:, :, 1]),
                        np.sin(descriptor[:, :, 0]) * np.sin(descriptor[:, :, 1]), np.cos(descriptor[:, :, 0])), axis=2)
        total += np.mean(np.sum(des[:, :, np.newaxis, :] * des[:, np.newaxis, :, :], axis=3))

    return total / batchs
