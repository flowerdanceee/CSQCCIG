import math
import numpy as np
# Pytorch imports
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.autograd as autograd
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from torch.optim import lr_scheduler as lr_s
import torch.nn.functional as F

trans = torchvision.transforms.Compose([
    torchvision.transforms.Resize(28),
    torchvision.transforms.ToTensor()])


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


def get_one_hot_tensor(tensor):
    one_hot_tensor = torch.zeros(tensor.size(0), 2)
    for i, num in enumerate(tensor):
        one_hot_tensor[i, num] = 1
    return one_hot_tensor


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


class Generator(nn.Module):
    # initializers
    def __init__(self, c=4):
        super(Generator, self).__init__()
        self.deconv1_1 = nn.ConvTranspose2d(8, 2 * c, 4, 1, 0)
        self.deconv1_1_bn = nn.BatchNorm2d(2 * c)
        self.deconv1_2 = nn.ConvTranspose2d(2, 2 * c, 4, 1, 0)
        self.deconv1_2_bn = nn.BatchNorm2d(2 * c)
        self.deconv2 = nn.ConvTranspose2d(4 * c, 2 * c, 3, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(2 * c)
        self.deconv3 = nn.ConvTranspose2d(2 * c, c, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(c)
        self.deconv4 = nn.ConvTranspose2d(c, 1, 4, 2, 1)

    # forward method
    def forward(self, input, label):
        x = F.relu(self.deconv1_1_bn(self.deconv1_1(input)))
        y = F.relu(self.deconv1_2_bn(self.deconv1_2(label)))
        x = torch.cat([x, y], 1)
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        # 7
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        # 14
        x = F.tanh(self.deconv4(x))
        # 28

        return x


# Enable CUDA device if available
device = torch.device("cpu")
lambda_gp = 10


def save_images(tensor, label):
    images_dir_0 = './CNN_fid/c=2/' + '/0/'
    images_dir_1 = './CNN_fid/c=2/' + '/1/'
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


discriminator = Discriminator().to(device)
generator = Generator().to(device)
total_params = sum(p.numel() for p in generator.parameters())
# total_params: 104*c*c+357*c+1
print(f"Total number of parameters: {total_params}")

# Optimisers
optD = optim.Adam(discriminator.parameters(), lr=0.003)
optG = optim.Adam(generator.parameters(), lr=0.0002)
scheduler = lr_s.StepLR(optG, step_size=20, gamma=0.4)

# Collect images for plotting later
test_label = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]).to(device)
test_label_tensor = get_one_hot_tensor(test_label)
test_label_tensor_g = torch.unsqueeze(test_label_tensor, -1)
test_label_tensor_g = torch.unsqueeze(test_label_tensor_g, -1)
D_loss = []

for epoch in range(1, 101):
    fixed_noise = torch.rand(10, 8, 1, 1, device=device) * math.pi / 2
    for (data, label) in train_loader:
        # Data for training the discriminator
        real_data = data.to(device)
        label_tensor = get_one_hot_tensor(label).to(device)
        label_tensor_g = torch.unsqueeze(label_tensor, -1)
        label_tensor_g = torch.unsqueeze(label_tensor_g, -1)
        real_labels = torch.full((label.size(0),), 1.0, dtype=torch.float, device=device)
        fake_labels = torch.full((label.size(0),), 0.0, dtype=torch.float, device=device)
        # Training the discriminator
        discriminator.zero_grad()
        noise = torch.rand(label.size(0), 8, 1, 1, device=device) * math.pi / 2
        fake_data = generator(noise, label_tensor_g).view(label.size(0), 1, 28, 28)
        outD_real = discriminator(real_data, label_tensor).view(-1)
        outD_fake = discriminator(fake_data.detach(), label_tensor.detach()).view(-1)
        gradient_penalty = compute_gradient_penalty(discriminator, real_data, fake_data, label_tensor)
        errD = -torch.mean(outD_real) + torch.mean(outD_fake) + lambda_gp * gradient_penalty
        # Propagate gradients
        errD.backward()
        optD.step()

        if epoch % 5 == 0:
            D_loss.append(-errD.item())
        # Training the generator
        if epoch % 5 == 0:
            fake_data = generator(noise, label_tensor_g).view(label.size(0), 1, 28, 28)
            outD_fake = discriminator(fake_data, label_tensor).view(-1)
            errG = -torch.mean(outD_fake)
            generator.zero_grad()
            errG.backward()
            optG.step()
            print(f'Epoch: {epoch}, Discriminator Loss: {errD:0.3f}, Generator Loss: {errG:0.3f}')

    scheduler.step()

d = np.array(D_loss)
np.save('CNN_loss/new.npy', d)

generator.eval()

label0 = torch.zeros((5000, 2)).to(device)
label0[:, 0] = 1
label0_g = torch.unsqueeze(label0, -1)
label0_g = torch.unsqueeze(label0_g, -1)
label1 = torch.zeros((5000, 2)).to(device)
label1[:, 1] = 1
label1_g = torch.unsqueeze(label1, -1)
label1_g = torch.unsqueeze(label1_g, -1)
fixed_noise = torch.rand(5000, 8, 1, 1, device=device) * math.pi / 2
images0 = generator(fixed_noise, label0_g).view(5000, 1, 28, 28)
save_images(images0, 0)
images1 = generator(fixed_noise, label1_g).view(5000, 1, 28, 28)
save_images(images1, 1)
