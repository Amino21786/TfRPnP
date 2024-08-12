import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import radon, iradon
from skimage.data import shepp_logan_phantom
from skimage.transform import resize
"""
n = 24

circle = False
n_angles = 60
theta = np.linspace(0., 180.,n_angles, endpoint=False)
sigma = 100e-1
size = (radon(np.zeros((n,n)), theta=theta, circle = circle)).shape
M = size[0]*size[1]
A = np.zeros((M, n**2))
for i in range(n**2):
    e = np.zeros((n**2,))
    e[i]=1
    e = np.reshape(e,(n,n))
    sinogram = radon(e, theta=theta, circle = circle)
    A[:,i] = np.reshape(sinogram[:], (M,))

# Load the Shepp-Logan phantom
phantom = resize(shepp_logan_phantom(), (n,n))

#ground truth
x = phantom.reshape(-1)

#b
y = np.reshape(A@x, (int(np.ceil(n*np.sqrt(2))), n_angles))
plt.title("Sinogram")
plt.imshow(y, cmap='gray')
plt.savefig("sinogram")
"""



import torch
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import torchvision
device = 'cuda' if torch.cuda.is_available() else 'cpu'

print(device)

torch.manual_seed(42)  # You can use any number here

strong_conv_constant = 0#1e-3
noise_level = 0.01
huber_const = 0.01


N = 100

n=96
images_size = (n, n)

# dataset = torch.load('celeba_dataset128.pt')
# print(dataset.shape)

dataset = torchvision.datasets.STL10('/local/scratch/public/hyt35/datasets/STL10', split='train', transform=torchvision.transforms.ToTensor(), folds=1, download=True)

dataset = torch.utils.data.Subset(dataset, list(range(N)))

dataloader = DataLoader(dataset, batch_size=N, shuffle=True)


def gaussian_kernel(size: int, sigma: float) -> torch.Tensor:
    """Creates a 2D Gaussian kernel."""
    x = torch.arange(size).float() - size // 2
    x = x.view(-1, 1)
    y = torch.arange(size).float() - size // 2
    y = y.view(1, -1)
    kernel_2d = torch.exp(-0.5 * (x*2 + y2) / sigma*2)
    kernel_2d = kernel_2d / kernel_2d.sum()
    return kernel_2d.unsqueeze(0).unsqueeze(0)

size = 5
sigma = 1.5
kernel = gaussian_kernel(size, sigma).to(device)


def convolution(kernel, image, same=True, groups=1):
    """
    Perform Mathematically correct 2D convolution using PyTorch's built-in function.

    Args:
        kernel (torch.Tensor): Convolution kernel of shape (out_channels, in_channels, kernel_height, kernel_width).
        image (torch.Tensor): Input image tensor of shape (batch_size, channels, height, width).

    Returns:
        torch.Tensor: Output tensor after 2D convolution with the kernel. Shape is same as input image.
    """
    ## shape has to be [N, C, H, W]:
    if len(image.shape) == 2:
        image = image.unsqueeze(0).unsqueeze(0)
    elif len(image.shape) == 3:
        image = image.unsqueeze(0)
    if len(kernel.shape) == 2:
        kernel = kernel.unsqueeze(0).unsqueeze(0)
    elif len(kernel.shape) == 3:
        kernel = kernel.unsqueeze(0)
    if len(image.shape) == 4 and len(kernel.shape) == 4:
        if same:
            return torch.flip(F.conv2d(torch.flip(kernel, dims=(2,3)), image, padding='same', groups=groups), dims=(2,3)) ## equal to torch.flip(F.conv2d( torch.flip(image, dims=(2,3)),kernel, padding='same'), dims=(2,3))
        else:
            return torch.flip(F.conv2d(torch.flip(kernel, dims=(2,3)), image, padding=image.shape[-1]-1, groups=groups), dims=(2,3)) ## equal to torch.flip(F.conv2d( torch.flip(image, dims=(2,3)),kernel, padding='same'), dims=(2,3))
    else:
        raise ValueError('Image shape is not correct')
    
forward_operator = lambda x: convolution(x, kernel)
adjoint_operator = lambda x: convolution(x, kernel)

def init_recon_blur(x):
    return x

init_recon = init_recon_blur

operator_norm = 1.#power_method_square_func(lambda x: convolution(x, kernel), n, dtype=kernel.dtype, num_iters=10000, tol=0)


def huber(s: torch.Tensor, epsilon: float = 0.01) -> torch.Tensor:
    """Compute the Huber loss element-wise.

    Args:
    - s: The input tensor.
    - epsilon: The threshold value for the Huber loss. Defaults to 0.01.

    Returns:
    - The element-wise Huber loss tensor.

    Raises:
    - ValueError: If the input tensor is not a torch.Tensor.
    """
    if not isinstance(s, torch.Tensor):
        raise ValueError("Input 's' to function huber() must be a torch.Tensor")

    if epsilon <= 0:
        raise ValueError("'epsilon' in function huber() must be greater than zero.")

    return torch.where(
        torch.abs(s) <= epsilon,
        (0.5 * s ** 2) / epsilon,
        torch.abs(s) - 0.5 * epsilon
    )




def fast_huber_TV(x, alpha=1, delta=0.01):
    ## in a batch it just adds all individuals up
    tv_h = huber(x[:, :, 1:,:]-x[:, :, :-1,:], delta).sum()
    tv_w = huber(x[:, :, :,1:]-x[:, :, :,:-1], delta).sum()
    huber_tv = (tv_h+tv_w)
    return alpha*huber_tv

def fast_huber_grad(x, alpha=1, delta=0.01):
    # Extract height and width dimensions
    height, width = x.shape[2:]
    batch_size = x.shape[0]
    diff_y = torch.cat((x[:, :, 1:, :] - x[:, :, :-1, :], torch.zeros(batch_size, 1, 1, width, dtype=x.dtype).to(x.device)), dim=2)
    diff_x = torch.cat((x[:, :, :, 1:] - x[:, :, :, :-1], torch.zeros(batch_size, 1, height, 1, dtype=x.dtype).to(x.device)), dim=3)
    delta_ones = delta * torch.ones_like(diff_y)
    max_diff_delta1 = torch.max(torch.abs(diff_y), delta_ones)
    max_diff_delta2 = torch.max(torch.abs(diff_x), delta_ones)
    penult1 = diff_y / max_diff_delta1
    penult2 = diff_x / max_diff_delta2
    f1 = torch.cat((-penult1[:, :, 0, :].unsqueeze(-2), -penult1[:, :, 1:, :] + penult1[:, :, :-1, :]), dim=2)
    f2 = torch.cat((-penult2[:, :, :, 0].unsqueeze(-1), -penult2[:, :, :, 1:] + penult2[:, :, :, :-1]), dim=3)
    result = alpha * (f1 + f2)
    return result




def psnr(imgs1, imgs2):
    total_psnr = 0.0
    for i in range(len(imgs1)):
        mse = F.mse_loss(imgs1[i], imgs2[i])
        max_pixel = imgs2[i].max()  # Assuming pixel values are normalized between 0 and 1
        psnr = 20 * torch.log10(max_pixel) - 10 * torch.log10(mse)
        total_psnr += psnr.item()
    return total_psnr / len(imgs1)

def psnr_every_image(imgs1, imgs2):
    psnrs = []
    for i in range(len(imgs1)):
        mse = F.mse_loss(imgs1[i], imgs2[i])
        max_pixel = imgs2[i].max()  # Assuming pixel values are normalized between 0 and 1
        psnr = 20 * torch.log10(max_pixel) - 10 * torch.log10(mse)
        psnrs.append(psnr.item())
    return psnrs

if _name_ == '_main_':

        #### JUST GETTING VALIDATION DATA
        for i, data in enumerate(dataloader):
            try:
                xs, label = data
            except:
                xs = data
            xs = xs.to(device)
            xs = F.interpolate(xs, size=(n, n), mode='bilinear', align_corners=False)
            ## make images greyscale
            xs = xs.mean(dim=1, keepdim=True)
            
            clean_observations = forward_operator(xs)
            noisy_observations = clean_observations + (noise_level*torch.randn_like(clean_observations)*torch.mean(clean_observations.view(xs.shape[0], -1), dim=1).unsqueeze(1).unsqueeze(1).unsqueeze(1))#.double()

            for i in range(10):
                plt.imshow(noisy_observations[i].squeeze().cpu().numpy(), cmap='viridis')
                plt.title('Noisy Observation')
                plt.show()
            
            reg_const_list = [1e-5, 1e-4, 1e-3]
            
            psnrs = []
            max_psnrs = []


            for reg_const in reg_const_list:
                
                new_indiv_psnr_lst = []
                new_indiv_psnrs_every_image = []
                
                reg_L_part = reg_const*8/huber_const

                L = (operator_norm**2 + reg_L_part + strong_conv_constant)

                one_over_L = 1/L
                def data_fit(x, y):
                    return 0.5*torch.norm(forward_operator(x) - y)**2/x.shape[0]

                def reg_func(x):
                    return (reg_const*fast_huber_TV(x, delta=huber_const) + .5*strong_conv_constant*torch.norm(x)**2)/x.shape[0]

                def grad_reg_func_total(x):
                    return (reg_const*fast_huber_grad(x, delta=huber_const) + strong_conv_constant*x)#/x.shape[0]

                def grad_func(x, y):
                    return adjoint_operator(forward_operator(x) - y) + grad_reg_func_total(x)

                def objective_function(x, y):
                    return data_fit(x,y)  + reg_func(x)



                inpt = init_recon(noisy_observations)
                agd = [objective_function(inpt, noisy_observations).item()]

                inpt_find_min = inpt.clone()
                inpt_find_min_m1 = inpt.clone()
                told = 0
                tnew = 0

                imgs_agd = [inpt.clone()]
                N_images = 7

                fig, axs = plt.subplots(1, N_images, figsize=(N_images*5, 5))
                fig2, axs2 = plt.subplots(1, N_images, figsize=(N_images*5, 5))

                img_counter = 0
                for k in tqdm(range(201)):

                    tnew, told = (1 + np.sqrt(1 + 4*told**2))/2, tnew
                    alphat = (told - 1)/tnew
                    yk = inpt_find_min + alphat*(inpt_find_min - inpt_find_min_m1)
                    grad_heavy_ball_oneL = grad_func(yk, noisy_observations)
                    #grad_heavy_ball_oneL = grad_func(inpt_find_min, noisy_observations)
                    inpt_find_min, inpt_find_min_m1 = yk - one_over_L*grad_heavy_ball_oneL, inpt_find_min
                    agd.append(objective_function(inpt_find_min, noisy_observations).item())
                    new_indiv_psnr_lst.append(psnr(inpt_find_min, xs))
                    new_indiv_psnrs_every_image.append(psnr_every_image(inpt_find_min, xs))
                    if k %40== 0:
                            axs[img_counter].imshow(inpt_find_min[0].squeeze().cpu().numpy(), cmap='viridis')
                            axs[img_counter].axis('off')
                            axs2[img_counter].imshow(inpt_find_min[1].squeeze().cpu().numpy(), cmap='viridis')
                            axs2[img_counter].axis('off')
                            #plt.imshow(inpt_find_min[10,:,:,:].squeeze().cpu().numpy(), cmap='gray')
                            #plt.title(f'FISTA Reconstruction Iteration Validation{k}')
                            #plt.show()
                            img_counter += 1
                axs[img_counter].imshow(xs[0].squeeze().cpu().numpy(), cmap='viridis')
                axs[img_counter].set_title(f'Ground Truth')
                axs[img_counter].axis('off')
                axs2[img_counter].imshow(xs[1].squeeze().cpu().numpy(), cmap='viridis')
                axs2[img_counter].set_title(f'Ground Truth')
                axs2[img_counter].axis('off')

                plt.tight_layout()
                plt.show()

                approx_min = min(agd)
                plt.semilogy((-approx_min + np.array(agd))/agd[0])
                plt.title('AGD Objective Function ValueVAL')
                plt.show()
                print('Approx min vals', approx_min)
                approx_min_image = inpt_find_min.clone()

                plt.plot(new_indiv_psnr_lst)
                plt.title('PSNR Values for this reg const through iterations')
                plt.show()

                print('ABOVE WAS FOR REG CONST', reg_const, '\n\n\n\n')

                psnrs.append(psnr(approx_min_image, xs))
                max_psnrs.append(max(new_indiv_psnr_lst))

                max_psnr_for_indiv_images = np.array(new_indiv_psnrs_every_image).max(axis=0)
                index_each_max_attained = np.array(new_indiv_psnrs_every_image).argmax(axis=0)

                ## boxplot of index_each_max_attained:
                plt.boxplot(index_each_max_attained)
                plt.ylabel('Iteration Number')
                plt.title('Boxplot of Iteration Number at which Maximum PSNR is attained for each image')
                plt.show()

            plt.semilogx(reg_const_list, psnrs)
            plt.title('PSNR vs Regularisation Constant')
            plt.show()

            plt.semilogx(reg_const_list, max_psnrs)
            plt.title('Max PSNR vs Regularisation Constant')
            plt.show()

