import numpy as np
import matplotlib.pyplot as plt
import torch 


#CT Radon transform forward model and denoising functions
from skimage.transform import radon, resize, iradon
from skimage.data import shepp_logan_phantom
from skimage.restoration import denoise_tv_chambolle
import bm3d 


from sklearn.feature_extraction import image
import deepinv as deepinv
from deepinv.models import DnCNN, DRUNet
from deepinv.optim.data_fidelity import L2
from deepinv.optim.prior import PnP

# Test image
phantom = shepp_logan_phantom()
num_scans = 180
angles = np.linspace(0., 360, num_scans, endpoint=False)
phantom_resized = resize(phantom, (200, 200), mode='reflect')

max_pixel = np.amax(phantom)
print(max_pixel)

#Peak signal to noise ratio
def PSNR(original, img, max_value = max_pixel): 
    mse = np.mean((np.array(original, dtype=np.float32) - np.array(img, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    psnr = 20 * np.log10(max_value / (np.sqrt(mse)))
    return psnr


#Display image results
def display_images(image_list, image_titles, ground_truth):
    titles = []
    for img, title in zip(image_list, image_titles):
        
        #cur_min = np.round(np.amin(img), 1)
        #cur_max = np.round(np.amax(img), 1)
        #bounds = '{} to {}'.format(str(cur_min), str(cur_max))
        psnr = PSNR(ground_truth, img)
        titles.append(title + f' PSNR: {psnr:.2f} dB \n')
        

        #nrmse = pnpm.nrmse(img, ground_truth)
        #titles.append(title + ' [NRMSE: ' + str(nrmse) + ']')

    for img, title in zip(image_list, titles):
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4.5, 5))
        

        ax.imshow(img, cmap='gray')
        #pnpm.display_image(img, fig=fig, ax=ax, cmap='gray')

        plt.suptitle(title)
        plt.tight_layout()
        fig.show()

n = 32

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

x = phantom.reshape(-1)

plt.imshow(phantom, cmap='gray')
plt.title("Ground truth (Shepp-Logan Phantom)")

clean_sinogram_flattened = A@x.flatten()
b = clean_sinogram_flattened + np.random.normal(0,1,2760)

y = np.reshape(A@x, (int(np.ceil(n*np.sqrt(2))), n_angles))
s = np.reshape(b, (int(np.ceil(n*np.sqrt(2))), n_angles))

from tqdm import tqdm

def pnp_admm(A, b, x_ground_truth, denoiser, l = 20, niter = 20):
    """
    Alternating directions method of multiplers with different denoisers to use  
    """ 

    def apply_denoiser(x, method):
        """
        Denoiser options for the u step (replacing the proximal operator)
        """
        if method == 'tv':
            return denoise_tv_chambolle(x, weight = 0.3)
        elif method == 'bm3d':
            return bm3d(x, sigma_psd = 5.9)
        elif method == 'proximal':
            return np.sign(x)*np.maximum(0, x)
        """
        elif method == "DnCNN":

            return True
        elif method == "DRUNet":
            return True
        """

    def nrmse(x, x_ground_truth):
        """
        Normalised root mean square error between final prediction and ground truth image
        """
        return np.linalg.norm(x - x_ground_truth)/np.linalg.norm(x_ground_truth)

    
    #dimensions
    n = int(np.sqrt(A.shape[1]))
    #defining the 3 variables for use
    x = np.zeros_like(x_ground_truth)
    u = np.zeros_like(x_ground_truth)
    v = np.zeros_like(x_ground_truth)
    metric_list = []

    #PnP ADMM iteration performed on the 3 variables
    for i in tqdm(range(niter), desc = 'PnP ADMM iterations'):
        
        #FBP inversion
        measurement = np.reshape(b, (int(np.ceil(n*np.sqrt(2))), n_angles))
        correction = np.reshape(A@(l*(u - v)), (int(np.ceil(n*np.sqrt(2))), n_angles))
        fbp = measurement + correction
        x = iradon(fbp, theta= theta, filter_name='ramp', circle=False)
        #x = np.linalg.solve(A.T @ A + rho * np.eye(A.shape[1]), A.T @ b + rho * (u - v))

        # reshaping u and v for the denoiser (1D to 2D)
        u = u.reshape((n, n))
        v = v.reshape((n, n))
        u = apply_denoiser(x + v, denoiser)
        v += x - u

        #flattening u and v for the next iteration when applying FBP (2D to 1D)
        u = u.flatten()
        v = v.flatten()


        current_psnr = PSNR(x_ground_truth, x.flatten())
        #print("current PSNR:", current_psnr)

        #current_nrmse = nrmse(x.flatten(), x_ground_truth)
        metric_list.append(current_psnr)

        
    print("Final PSNR:", current_psnr)

    return x, metric_list

result_proximal, metrics_prox = pnp_admm(A, b, x, 'proximal', niter = 50)
display_images([result_proximal], ['Proximal'], phantom)

def comparison_plot(norm_res, method, ytitle='PSNR', color='k'):
    K = len(norm_res)

    plt.plot(range(1, len(norm_res) + 1),norm_res, color=color, label=method)
    k = np.arange(K)

    plt.ylabel(ytitle)
    plt.xlabel('number of iterations: k')
    plt.legend()
    plt.grid(True)

comparison_plot(metrics_prox, 'Proximal operator', color='purple')