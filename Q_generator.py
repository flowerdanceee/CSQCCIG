import math
import pennylane as qml
from tqdm import tqdm
from PIL import Image
import numpy as np

# Pytorch imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader, Dataset
import torch.autograd as autograd
from torch.optim import lr_scheduler as lr_s

image_size = 28
trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor()])
if not os.path.exists('./Q_model/new'):
    os.makedirs('./Q_model/new')


def get_one_hot_tensor(tensor):
    one_hot_tensor = torch.zeros(tensor.size(0), 2)
    for i, num in enumerate(tensor):
        one_hot_tensor[i, num] = 1
    return one_hot_tensor


def compute_gradient_penalty(critic, real_samples, fake_samples, label_tensor):
    """Calculates the gradient penalty loss for WGAN GP"""
    batch_size, C1, W1, H1 = real_samples.shape
    epsilon1 = torch.rand(batch_size, 1, 1, 1).repeat(1, C1, W1, H1)
    interpolated_images = (epsilon1 * real_samples + ((1 - epsilon1) * fake_samples))
    interpolated_scores = critic(interpolated_images, label_tensor)

    # Get gradient w.r.t. interpolates
    gradients = autograd.grad(
        inputs=interpolated_images,
        outputs=interpolated_scores,
        grad_outputs=torch.ones_like(interpolated_scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    gradients = gradients.view(gradients.shape[0], -1)
    gradient_penalty = torch.mean((1. - torch.sqrt(1e-8 + torch.sum(gradients ** 2, dim=1))) ** 2)
    return gradient_penalty


class Mydata(Dataset):
    def __init__(self, root, transforms=trans):
        self.root_dir = root
        self.transform = transforms
        self.label_list = os.listdir(self.root_dir)

    def __getitem__(self, idx):
        (a, b) = divmod(idx, 100)
        label = self.label_list[a]
        img_list = os.listdir(os.path.join(self.root_dir, label))
        img_name = img_list[b]
        img_item_path = os.path.join(self.root_dir, label, img_name)
        img = Image.open(img_item_path)
        img = self.transform(img)
        return img, int(label)

    def __len__(self):
        return 200


# Mnist
data = Mydata('mnist-data', transforms=trans)
train_loader = DataLoader(dataset=data, batch_size=16, shuffle=True, pin_memory=True)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            # Inputs to first hidden layer (num_input_features -> 64)
            nn.Linear(784 + 2, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            # First hidden layer (64 -> 16)
            nn.Linear(1024, 256),
            nn.ReLU(),
            # Second hidden layer (16 -> output)
            nn.Linear(256, 1)
        )

    def forward(self, x, label):
        x = x.view(x.size(0), -1)
        x = torch.cat((x, label), 1)
        x = self.model(x)
        return x


# Quantum variables
n_c_qubits = 1  # Number of ancillary qubits / N_A
n_a_qubits = 1
q_depth = 6  # Depth of the parameterised quantum circuit / D
n_generators = 8  # Number of sub-generators for the patch method / N_G
n_qubits = math.ceil(
    math.log(image_size ** 2 // n_generators, 2)) + n_a_qubits + n_c_qubits  # Total number of qubits / N
print(n_qubits)
# Quantum simulator
dev = qml.device("lightning.qubit", wires=n_qubits)
# Enable CUDA device if available
device = torch.device("cpu")


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


def save_images(epoch, tensor, label):
    images_dir_0 = './quantum/new/' + str(epoch) + '/0/'
    images_dir_1 = './quantum/new/' + str(epoch) + '/1/'
    if not os.path.exists(images_dir_0):
        os.makedirs(images_dir_0)
    if not os.path.exists(images_dir_1):
        os.makedirs(images_dir_1)

    for i in range(tensor.shape[0]):
        if i < 5:
            torchvision.utils.save_image(tensor[i], images_dir_0 + str(i) + '_' + str(label[i].item()) + '.png')
        else:
            torchvision.utils.save_image(tensor[i], images_dir_1 + str(i) + '_' + str(label[i].item()) + '.png')


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


lrG = 0.03  # Learning rate for the generator
lrD = 0.0002  # Learning rate for the discriminator
lambda_gp = 10
discriminator = Discriminator().to(device)
generator = PatchQuantumGenerator(n_generators, image_size).to(device)
D_loss = []

# Optimisers
optD = optim.Adam(discriminator.parameters(), lr=lrD)
optG = optim.Adam(generator.parameters(), lr=lrG)
scheduler = lr_s.StepLR(optD, step_size=20, gamma=0.2)

# Collect images for plotting later
test_label = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).to(device)
for epoch in tqdm(range(1, 101)):
    fixed_noise = torch.rand(10, n_qubits - n_c_qubits, device=device) * math.pi / 2
    for (data, label) in train_loader:
        # Data for training the discriminator
        real_data = data.to(device)
        label_tensor = get_one_hot_tensor(label).to(device)
        real_labels = torch.full((label.size(0),), 1.0, dtype=torch.float, device=device)
        fake_labels = torch.full((label.size(0),), 0.0, dtype=torch.float, device=device)
        # Training the discriminator
        discriminator.zero_grad()
        noise = torch.rand(label.size(0), n_qubits - n_c_qubits, device=device) * math.pi / 2
        fake_data = generator(label, noise).view(label.size(0), 1, image_size, image_size)
        outD_real = discriminator(real_data, label_tensor)
        outD_fake = discriminator(fake_data.detach(), label_tensor.detach())
        # Propagate gradients
        wasserstein_distance = torch.mean(outD_real) - torch.mean(outD_fake)
        gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, label_tensor)
        errD = -torch.mean(outD_real) + torch.mean(outD_fake) + lambda_gp * gradient_penalty
        errD.backward()
        optD.step()
        generator.zero_grad()
        print(f'Epoch: {epoch}, Discriminator Loss: {errD:0.3f}')
        if epoch % 5 == 0:
            D_loss.append(-errD.item())

        # Training the generator
        if epoch % 5 == 0:
            fake_data = generator(label, noise).view(label.size(0), 1, image_size, image_size)
            outD_fake = discriminator(fake_data, label_tensor)
            errG = -torch.mean(outD_fake)
            generator.zero_grad()
            errG.backward()
            optG.step()
            print(f'Epoch: {epoch}, Generator Loss: {errG:0.3f}')
    if epoch % 5 == 0:
        test_images = generator(test_label, fixed_noise).view(10, 1, image_size, image_size)
        save_images(epoch, test_images, test_label)
        torch.save(generator.state_dict(), './Q_model/new/model_parameters_' + str(epoch) + '.pth')

    scheduler.step()

d = np.array(D_loss)
np.save('Q_loss/new.npy', d)
