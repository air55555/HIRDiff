import random
import numpy as np
import torch
from matplotlib import pyplot as plt
import torch
import numpy as np
import  spectral
from spectral import envi




def display_envi(file_path):
    """
    Displays an ENVI file as an image.

    Parameters:
    file_path (str): Path to the ENVI file (.hdr or .dat file).

    Returns:
    None
    """
    try:
        # Load the ENVI image
        img = spectral.open_image(file_path)

        # Display the first band or RGB (if available)
        if img.nbands == 1:
            spectral.imshow(img)
        elif img.nbands >= 3:
            spectral.imshow(img, bands=(0, 1, 2))  # Adjust bands as needed for RGB
        else:
            print("Unable to determine displayable bands.")

        plt.show()
    except Exception as e:
        print(f"Error displaying ENVI file: {e}")


# Example usage
#display_envi('example_file.hdr')


def save_envi(tensor,fname):
    # Example tensor
    #tensor = torch.randn(1, 191, 256, 256)  # Replace with your actual tensor

    # Remove batch dimension and permute to match ENVI format (bands, rows, cols)
    hyperspectral_data = tensor.squeeze(0).numpy()  # Shape: [191, 256, 256]

    # Save as ENVI file
    output_file = fname
    metadata = {
        "description": "Sample hyperspectral image",
        "lines": 256,
        "samples": 256,
        "bands": 191,
        "interleave": "bil",  # Band-interleaved by line
        "data type": 4,       # Floating point (32-bit)
        "byte order": 0,      # Little-endian
    }

    # Save the image
    envi.save_image(f"{output_file}.hdr", hyperspectral_data, dtype=np.float32, metadata=metadata,force=True)

def my_diff(x):
    diff_1, diff_2, diff_3 = torch.zeros_like(x), torch.zeros_like(x), torch.zeros_like(x)
    diff_1[:, :-1, ...] = x[:, :-1, ...] - x[:, 1:, ...]
    diff_1[:, -1, ...] = x[:, -1, ...] - x[:, 0, ...]
    diff_2[:, :, :-1, ...] = x[:, :, :-1, ...] - x[:, :, 1:, ...]
    diff_2[:, :, -1, ...] = x[:, :, -1, :] - x[:, :, 0, ...]
    diff_3[..., :-1] = x[..., :-1] - x[..., 1:]
    diff_3[..., -1] = x[..., -1] - x[..., 0]
    return diff_1, diff_2, diff_3

def my_svd(x, rank):
    B, C, H, W = x.shape
    u, s, v = torch.svd(x.reshape(B, C, -1).permute(0, 2, 1))
    A = v[:, :, :rank]
    M = u[:, :, :rank] @ torch.diag_embed(s[:, :rank])
    M = M.permute(0, 2, 1).reshape(B, -1, H, W)
    x_lr = torch.einsum('bcr, brhw->bchw', A, M)
    return x_lr, A, M


def worker_init_fn(worker_id):
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def minmax_normalize(array):
    array = array.astype(np.float)
    amin = np.min(array)
    amax = np.max(array)
    return (array - amin) / (amax - amin)

def torch2numpy(hsi, use_2dconv):
    if use_2dconv:
        R_hsi = hsi.data[0].cpu().numpy().transpose((1, 2, 0))
    else:
        R_hsi = hsi.data[0].cpu().numpy()[0, ...].transpose((1, 2, 0))
    return R_hsi





def seed_everywhere(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.