import os
import numpy as np
import json
import math
from PyAstronomy import pyasl
from multiprocessing import Pool, Process, Queue

import torch
import torch.nn as nn
import torch.optim as optim

an = pyasl.AtomicNo()

with open('output.json', 'r') as f:
    qm9 = json.load(f)

epsilon = 1e-15


def rotation_matrix(raxis='z', theta=0):
    dim = theta.shape[0]
    rmatrix = torch.zeros((dim, dim, 3, 3))
    if raxis == 'x':
        rmatrix[:, :, 0, 0] = 1
        rmatrix[:, :, 1, 1] = torch.cos(theta)
        rmatrix[:, :, 1, 2] = -torch.sin(theta)
        rmatrix[:, :, 2, 1] = torch.sin(theta)
        rmatrix[:, :, 2, 2] = torch.cos(theta)
        return rmatrix
    if raxis == 'y':
        rmatrix[:, :, 0, 0] = torch.cos(theta)
        rmatrix[:, :, 0, 2] = torch.sin(theta)
        rmatrix[:, :, 1, 1] = 1
        rmatrix[:, :, 2, 0] = -torch.sin(theta)
        rmatrix[:, :, 2, 2] = torch.cos(theta)
        return rmatrix
    if raxis == 'z':
        rmatrix[:, :, 0, 0] = torch.cos(theta)
        rmatrix[:, :, 0, 1] = -torch.sin(theta)
        rmatrix[:, :, 1, 0] = torch.sin(theta)
        rmatrix[:, :, 1, 1] = torch.cos(theta)
        rmatrix[:, :, 2, 2] = 1
        return rmatrix


def Set_center(molecule, set_axis=False):
    cord = torch.from_numpy(np.array(molecule["coordinates"])[:, 1:].astype('float32'))
    atomic_num = torch.zeros(len(cord[:, 0]))
    for i in range(len(cord[:, 0])):  # 원자 개수 n
        atomic_num[i] = an.getAtomicNo(molecule["coordinates"][i][0])
    distance_martix = torch.sqrt(torch.sum(torch.pow(cord[None, :, :] - cord[:, None, :], 2), axis=2))
    # (n,n)
    bloch_cord = (cord[:, None, :] - cord[None, :, :] + epsilon) / (distance_martix[:, :, None] + epsilon)  # n,n,3
    if set_axis == False:
        return bloch_cord, distance_martix, atomic_num, molecule['energy_U0']
    # (n,n,3)
    elif set_axis == True:  ##가장 가까운 원자를 z축 위에, 두번쨰로 가짜운 원자는 xz 평면 위에 올리기(rotation sym 맞추려고 강제)
        bloch_theta = torch.arccos(bloch_cord[:, :, 2])
        # (n,n)
        bloch_pi = torch.atan2(bloch_cord[:, :, 1], bloch_cord[:, :, 0])
        # (n,n)
        min_point = torch.argsort(distance_martix, axis=1)[:, :3]

        theta1 = torch.tensor([bloch_theta[i, x] for i, x in enumerate(min_point[:, 1])])
        theta2 = torch.tensor([bloch_theta[i, x] for i, x in enumerate(min_point[:, 2])])
        pi1 = torch.tensor([bloch_pi[i, x] for i, x in enumerate(min_point[:, 1])])
        pi2 = torch.tensor([bloch_pi[i, x] for i, x in enumerate(min_point[:, 2])])
        pi_prime = torch.atan2(torch.sin(theta2) * torch.sin(pi2 - pi1),
                               torch.cos(theta1) * torch.sin(theta2) * torch.cos(pi2 - pi1) - torch.sin(
                                   theta1) * torch.cos(theta2))
        rot3 = rotation_matrix('z', -pi_prime[:, None])
        rot2 = rotation_matrix('y', (-theta1)[:, None])
        rot1 = rotation_matrix('z', (-pi1)[:, None])
        sym_bloch_cord = torch.einsum('abij,abjk,abkl,abl-> abi', rot3, rot2, rot1, bloch_cord)
        sym_bloch_cord[(torch.abs(sym_bloch_cord) < epsilon)] = 0
        for i in range(len(sym_bloch_cord[:, 0, 0])):
            sym_bloch_cord[i, i, :] = 0
        return sym_bloch_cord, distance_martix, atomic_num, molecule['energy_U0']


def Cutoff_function(distance, cutoff_radius=5):
    return 0.5 * (torch.cos(math.pi * distance / cutoff_radius) + 1)


def calqubit(theta, phi):
    return torch.stack((torch.cos(theta / 2), torch.sin(theta / 2) * torch.exp(1j * phi)), axis=-1)


def returnangle(qubit):
    the = 2 * torch.atan2(torch.abs(qubit[:, :, 1]), torch.real(qubit[:, :, 0]))
    ph = torch.atan2(qubit[:, :, 1].imag, qubit[:, :, 1].real)
    return (torch.stack((the, ph), axis=2))


def Cal_descriptor(cord, distance_matrix, classic=False, desnet=None, classic_parameter=None, weigthed=False,
                   atomic_num=None, cutoff_radius=5, halve=False):
    if classic:
        descriptor = calqubit(torch.arccos(cord[:, :, 2]), torch.atan2(cord[:, :, 1], cord[:, :, 0])) * Cutoff_function(
            distance_matrix[:, :, None], cutoff_radius=cutoff_radius)
        descriptor = descriptor[:, None, :, :] * torch.exp(-classic_parameter[None, :, None, None, 0] * torch.pow(
            distance_matrix[:, None, :, None] - classic_parameter[None, :, None, None, 1], 2))
        if weigthed:
            atomic_weight = torch.zeros((len(atomic_num), len(atomic_num)))
            for i in range(len(atomic_num)):
                for j in range(len(atomic_num)):
                    atomic_weight[i, j] = atomic_num[i] * atomic_num[j]
            descriptor = descriptor * atomic_weight[:, None, :, None]
    else:
        descriptor = calqubit(torch.arccos(cord[:, :, 2]), torch.atan2(cord[:, :, 1], cord[:, :, 0]))
        descriptor = descriptor[:, None, :, :] * desnet(distance_matrix, atomic_num)[None, :, None, None]
    descriptor = torch.sum(descriptor, axis=2)
    descript_size = torch.sum(descriptor * descriptor.conj(), axis=2)
    descriptor = descriptor / torch.sqrt(descript_size[:, :, None])

    return returnangle(descriptor), descript_size


def AtomLoader1(sampler=None, idx=None, numb=1, classic=False, desnet=None, classic_parameter=None,
                weigthed=False, cutoff_radius=5, halve=False):
    if sampler is None:
        atomloader = {}
        for i, x in enumerate(idx):
            sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[x.astype(str)])
            descriptor, descript_size = Cal_descriptor(
                cord=sym_bloch_cord,
                distance_matrix=distance_martix,
                classic=classic,
                desnet=desnet,
                classic_parameter=classic_parameter,
                weigthed=weigthed,
                atomic_num=atomic_num,
                cutoff_radius=cutoff_radius,
                halve=halve
            )
            atomloader[i] = {
                'ground_energy': ground_energy,
                'descriptor': torch.tensor(descriptor, requires_grad=True),
                'descriptor_size': torch.tensor(descript_size, requires_grad=True),
                'atomic_number': qm9[x]['n_atoms']
            }
        return atomloader
    if sampler == 'random':
        idx = (torch.randint(133885, size=(numb,)) + 1)
        atomloader = {}
        for i, x in enumerate(idx):
            sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[str(x)])
            descriptor, descript_size = Cal_descriptor(
                cord=sym_bloch_cord,
                distance_matrix=distance_martix,
                classic=classic,
                desnet=desnet,
                classic_parameter=classic_parameter,
                weigthed=weigthed,
                atomic_num=atomic_num,
                cutoff_radius=cutoff_radius,
                halve=halve
            )
            atomloader[i] = {
                'ground_energy': ground_energy,
                'descriptor': torch.tensor(descriptor, requires_grad=True),
                'descriptor_size': torch.tensor(descript_size, requires_grad=True),
                'atomic_number': qm9[str(x)]['n_atoms']
            }
        return atomloader


def AtomLoader2(sampler=None, idx=None, epochs=1, batchs=1, classic=False, desnet=None, classic_parameter=None,
                weigthed=False, cutoff_radius=5, halve=False):
    if sampler is None:
        atomloader = {}
        for i, x in enumerate(idx):
            sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[x.astype(str)])
            descriptor, descript_size = Cal_descriptor(
                cord=sym_bloch_cord,
                distance_matrix=distance_martix,
                classic=classic,
                desnet=desnet,
                classic_parameter=classic_parameter,
                weigthed=weigthed,
                atomic_num=atomic_num,
                cutoff_radius=cutoff_radius,
                halve=halve
            )
            atomloader[i] = {
                'ground_energy': ground_energy,
                'descriptor': torch.tensor(descriptor, requires_grad=True),
                'descriptor_size': torch.tensor(descript_size, requires_grad=True),
                'atomic_number': qm9[x]['n_atoms']
            }
        return atomloader
    if sampler == 'random':
        idx = (torch.randint(133885, size=(epochs * batchs,)) + 1)
        atomloader = {}
        k = 0
        for i in range(epochs):
            atomload = {}
            for j in range(batchs):
                sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[str(idx[k].item())])
                descriptor, descript_size = Cal_descriptor(
                    cord=sym_bloch_cord,
                    distance_matrix=distance_martix,
                    classic=classic,
                    desnet=desnet,
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
                    'atomic_number': qm9[str(idx[k].item())]['n_atoms']
                }
                k += 1
            atomloader[i] = atomload
        return atomloader


num_cores = 24


def paraLoader(classic_parameter=None, weigthed=False, cutoff_radius=5):
    idx = (torch.randint(133885, size=(133885,)) + 1)
    array_split = torch.array_split(torch.arange(133885) + 1, num_cores)
    total = 0

    def caldot(x, q):
        out = 0
        for i, y in enumerate(array_split[x]):
            sym_bloch_cord, distance_martix, atomic_num, ground_energy = Set_center(qm9[str(y)])
            descriptor, descript_size = Cal_descriptor(
                cord=sym_bloch_cord,
                distance_matrix=distance_martix,
                classic=True,
                classic_parameter=classic_parameter,
                weigthed=weigthed,
                atomic_num=atomic_num,
                cutoff_radius=cutoff_radius,
            )
            des = torch.tensor([torch.sin(descriptor[:, :, 0]) * torch.cos(descriptor[:, :, 1]),
                                torch.sin(descriptor[:, :, 0]) * torch.sin(descriptor[:, :, 1]),
                                torch.cos(descriptor[:, :, 0])])
            out += torch.sum(des[:, :, None, :] * des[:, None, :, :])
        q.put(out)

    q = Queue()
    for i in range(num_cores):
        proc = Process(target=caldot, args=(i, q))
        proc.start()
    for i in range(num_cores):
        total = total + q.get()

    return total
