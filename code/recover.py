
import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
from skimage.transform import resize
from skimage.data import chelsea

torch.manual_seed(0);
plt.rcParams["font.size"] = "14"
plt.rcParams['toolbar'] = 'None'
plt.ion()

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu");
#use GPU
print(device)

nxd = 128
chelseaimage = chelsea() #cat image
true_object_np = 100.0 * resize(chelseaimage[10:299, 110:399, 2], (nxd, nxd), anti_aliasing=False)

fig1, axs1 = plt.subplots(2, 3, figsize=(20, 12))
plt.tight_layout()
#fig1.canvas.manager.window.move(0, 0)

axs1[0, 2].imshow(true_object_np, cmap='Greys_r');
axs1[0, 2].set_axis_off();

# CNN class set up
class CNN_configurable(nn.Module):
    def __init__(self, n_lay, n_chan, ksize):  # Corrected method name
        super(CNN_configurable, self).__init__()  # Corrected superclass call
        pd = int(ksize / 2)
        layers = [nn.Conv2d(1, n_chan, ksize, padding=pd), nn.PReLU()]
        for _ in range(n_lay):
            layers.append(nn.Conv2d(n_chan, n_chan, ksize, padding=pd));
            layers.append(nn.PReLU())
        layers.append(nn.Conv2d(n_chan, 1, ksize, padding=pd));
        layers.append(nn.PReLU())

        self.deep_net = nn.Sequential(*layers)

    def forward(self, x):  # Corrected method signature
        return torch.squeeze(self.deep_net(x.unsqueeze(0).unsqueeze(0)))


cnn = CNN_configurable(32,nxd,3).to(device) #creating a CNN object from the class

input_image = torch.rand(nxd, nxd).to(device)


# torch to numpy converter
def torch_to_np(torch_array):
    return np.squeeze(torch_array.detach().cpu().numpy())

# numpy to torch converter
def np_to_torch(np_array):
    return torch.from_numpy(np_array).float()  # Corrected method call

true_object_torch = np_to_torch(true_object_np).to(device)

# noisy example
measured_data = torch.poisson(true_object_torch)
#gap example
mask_image = torch.ones_like(measured_data)
mask_image[int(0.65 * nxd):int(0.85 * nxd), int(0.65 * nxd):int(0.85 * nxd)] = 0
mask_image[int(0.15 * nxd):int(0.25 * nxd), int(0.15 * nxd):int(0.35 * nxd)] = 0
#currupted data
measured_data = measured_data * mask_image  # Corrected variable name

axs1[0, 2].imshow(torch_to_np(true_object_torch), cmap='Greys_r');
axs1[0, 2].set_title('TRUE');
axs1[0, 2].set_axis_off();
axs1[0, 1].imshow(torch_to_np(measured_data), cmap='Greys_r');
axs1[0, 1].set_title('DATA');
axs1[0, 1].set_axis_off();
axs1[1, 0].imshow(torch_to_np(input_image), cmap='Greys_r');
axs1[1, 0].set_title('Input image %d x %d' % (nxd, nxd)); # Corrected method call and text
axs1[1, 0].set_axis_off();

cv2.waitKey(10000)


def nrmse_fn(recon, reference):  # Corrected parameter name
    num = (reference - recon) ** 2;
    denom = reference ** 2;
    return 100.0 * torch.mean(num) * 0.5 / torch.mean(denom) * 0.5

# train the network
optimizer = torch.optim.Adam(cnn.parameters(), lr=1e-4)  # Corrected variable name and learning rate
train_loss = list();
nrmse_list = list();
best_nrmse = 10e9

for ep in range(1000000 + 1):

    optimizer.zero_grad() # steup the gradients to zero
    output_image = cnn(input_image)

    loss = nrmse_fn(output_image * mask_image, measured_data * mask_image) # train on masked data

    train_loss.append(loss.item())
    loss.backward()  # Find the gradient
    optimizer.step()  # Does the update

    nrmse = nrmse_fn(output_image, true_object_torch)  # Evaluate error wrt true image overall
    nrmse_list.append(nrmse.item())
    if nrmse < best_nrmse or ep == 0:
        best_recon = output_image;
        best_nrmse = nrmse;
        best_ep = ep
        axs1[1, 2].cla()
        axs1[1, 2].imshow(torch_to_np(best_recon), cmap='Greys_r');
        axs1[1, 2].set_title('Best Recon %d, NRMSE = %.2f%%' % (best_ep, best_nrmse))
        axs1[1, 2].set_axis_off();

    if ep % 2 == 0:
        axs1[1, 1].cla();
        axs1[1, 1].imshow(torch_to_np(output_image), cmap='Greys_r');
        axs1[1, 1].set_title('Recon %d, NRMSE = %.2f%%' % (ep, nrmse));
        axs1[1, 1].set_axis_off();
        axs1[0, 0].cla();
        axs1[0, 0].plot(train_loss[-200:-1]);
        axs1[0, 0].plot(nrmse_list[-200:-1]);
        axs1[0, 0].set_title('NRMSE (%%), epoch %d' % ep);
        axs1[0, 0].legend(['Error wrt DATA', 'Error wrt TRUE'])
        cv2.waitKey(1)  # Allow time for update

