from QMLatomLoader import *
import pennylane as qml
from pennylane import numpy as np
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter

epochs = 2000
batchs = 32

n_qubit = 1
radius = 10

para = np.array(
    [[radius, 0],
     ]
)
para.requires_grad = False
print(para)


# para=np.array([[0,0.5],
#                [0.1,1],
#                [1,5]])

def AWS_provider(n_wires, qpu='sv1'):
    my_prefix = "qhack2022"
    if qpu == 'Aspen-11':
        device_arn = "arn:aws:braket:::device/qpu/rigetti/Aspen-11"
        my_bucket = "amazon-braket-6024fbc8bc7d"
    elif qpu == 'ionq':
        device_arn = "arn:aws:braket:::device/qpu/ionq/ionQdevice"
        my_bucket = "amazon-braket-6024fbc8bc7b"
    elif qpu == 'Aspen-M-1':
        device_arn = "arn:aws:braket:us-west-1::device/qpu/rigetti/Aspen-M-1"
        my_bucket = "amazon-braket-6024fbc8bc7d"
    else:
        device_arn = "arn:aws:braket:::device/quantum-simulator/amazon/sv1"
        my_bucket = "amazon-braket-6024fbc8bc7b"
    s3_folder = (my_bucket, my_prefix)

    dev_remote = qml.device(
        "braket.aws.qubit",
        device_arn=device_arn,
        wires=n_wires,
        s3_destination_folder=s3_folder,
        parallel=True,
    )

    return dev_remote


# dev = qml.device("default.qubit", wires=n_qubit)
dev = AWS_provider(n_qubit, qpu='Aspen-M-1')


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

if __name__ == '__main__':
    writer = SummaryWriter()
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
            cutoff_radius=radius,
            set_axis=True
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


            params, loss = opt.step_and_cost(losses, params)
            writer.add_scalar(f'Batch={batchs}/Loss', loss, i)
            losslist.append(loss.item())
    else:
        loadatom = AtomLoader2(
            sampler='random',
            epochs=epochs,
            batchs=batchs,
            classic=True,
            classic_parameter=para,
            weigthed=True,
            cutoff_radius=radius,
            set_axis=True
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


            params, loss = opt.step_and_cost(losses, params)
            writer.add_scalar(f'Batch={batchs}/Loss', loss, i)
            losslist.append(loss.item())

    print(para)
    print(losslist)
    print(params)
    paradict = {"descriptor_para": para.tolist(), "hamiltoninan_para": params.tolist(), "loss": losslist}
    with open('paradict.json', 'w') as fp:
        json.dump(paradict, fp)
