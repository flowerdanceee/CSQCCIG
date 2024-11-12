import torch
import torch.nn as nn
import math
import pennylane as qml
import os
import torchvision

image_size = 28
n_c_qubits = 1  # Number of ancillary qubits / N_A
n_a_qubits = 1
q_depth = 6  # Depth of the parameterised quantum circuit / D
n_generators = 8  # Number of sub-generators for the patch method / N_G
n_qubits = math.ceil(
    math.log(image_size ** 2 // n_generators, 2)) + n_a_qubits + n_c_qubits  # Total number of qubits / N
device = torch.device("cpu")
dev = qml.device("lightning.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def quantum_circuit(label, noise, weights):
    if label == torch.tensor(1):
        qml.PauliX(wires=0)

    for i in range(n_qubits - n_c_qubits):
        qml.RY(noise[i], wires=i + n_c_qubits)
        # Repeated layer
    for i in range(q_depth):
        # Parameterised layer
        for y in range(n_qubits - n_c_qubits):
            qml.Rot(*weights[i, y], wires=y + n_c_qubits)

        for y in range(n_qubits - n_c_qubits - 1):
            qml.CNOT(wires=[0, y + 1])
            qml.CNOT(wires=[y + 1, y + 2])
        qml.CNOT(wires=[0, n_qubits - 1])
        qml.CNOT(wires=[n_qubits - 1, 1])

    return qml.probs(wires=list(range(n_c_qubits, n_qubits)))


def partial_measure(label, noise, weights):
    # Non-linear Transform
    probs = quantum_circuit(label, noise, weights)
    # Post-Processing
    probsgiven0 = probs[:(2 ** (n_qubits - n_a_qubits - n_c_qubits))]
    probsgiven0 /= torch.sum(probs)

    # Post-Processing
    probsgiven = probsgiven0 / torch.max(probsgiven0)
    return probsgiven


class PatchQuantumGenerator(nn.Module):
    """Quantum generator class for the patch method"""

    def __init__(self, n_generators, image_size, q_delta=1):
        """
        Args:
            n_generators (int): Number of sub-generators to be used in the patch method.
            q_delta (float, optional): Spread of the random distribution for parameter initialisation.
        """

        super().__init__()

        self.q_params = nn.ParameterList(
            [
                nn.Parameter(q_delta * torch.rand((q_depth, (n_qubits - n_c_qubits), 3)),
                             requires_grad=True)
                for _ in range(n_generators)
            ]
        )
        self.n_generators = n_generators
        self.image_size = image_size
        self.pixels_per_patch = self.image_size ** 2 // n_generators

    def forward(self, label, x):

        # Create a Tensor to 'catch' a batch of images from the for loop. x.size(0) is the batch size.
        images = torch.Tensor(x.size(0), 0).to(device)

        # Iterate over all sub-generators
        for params in self.q_params:

            # Create a Tensor to 'catch' a batch of the patches from a single sub-generator
            patches = torch.Tensor(0, self.pixels_per_patch).to(device)
            for index in range(x.size(0)):
                q_out = partial_measure(label[index], x[index], params).float().unsqueeze(0).to(
                    device)
                q_out = q_out[:, :self.pixels_per_patch]
                patches = torch.cat((patches, q_out))

            # Each batch of patches is concatenated with each other to create a batch of images
            images = torch.cat((images, patches), 1)

        return images


def save_images(tensor, label):
    images_dir_0 = './quantum_fid/P=8,L=6/' + '/0/'
    images_dir_1 = './quantum_fid/P=8,L=6/' + '/1/'
    if not os.path.exists(images_dir_0):
        os.makedirs(images_dir_0)
    if not os.path.exists(images_dir_1):
        os.makedirs(images_dir_1)
    if label == 0:
        for i in range(tensor.shape[0]):
            torchvision.utils.save_image(tensor[i], images_dir_0 + str(i) + '.png')
    else:
        for i in range(tensor.shape[0]):
            torchvision.utils.save_image(tensor[i], images_dir_1 + str(i) + '.png')


generator = PatchQuantumGenerator(n_generators, image_size).to(device)
generator_state_dict = torch.load('Q_model/P=8,L=6/model_parameters_100.pth')
generator.load_state_dict(generator_state_dict)
generator.eval()

label0 = torch.zeros(5000).to(device)
label1 = torch.ones(5000).to(device)
fixed_noise = torch.rand(5000, n_qubits - n_c_qubits, device=device) * math.pi / 2
images0 = generator(label0, fixed_noise).view(5000, 1, image_size, image_size)
save_images(images0, 0)
images1 = generator(label1, fixed_noise).view(5000, 1, image_size, image_size)
save_images(images1, 1)
