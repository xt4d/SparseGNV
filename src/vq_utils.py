import numpy as np
from PIL import Image
from .utils import instantiate_from_config

import torch
from omegaconf import OmegaConf


def as_png(x):
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = (x+1.0)*127.5
    x = x.clip(0, 255).astype(np.uint8)

    return Image.fromarray(x)


@torch.no_grad()
def encode_to_c(vqmodel, c):
    quant_c, _, info = vqmodel.encode(c)
    indices = info[2].view(quant_c.shape[0], -1)
    return quant_c, indices


@torch.no_grad()
def decode_to_img(vqmodel, index, zshape):
    bhwc = (zshape[0],zshape[2],zshape[3],zshape[1])
    quant_z = vqmodel.quantize.get_codebook_entry(
        index.reshape(-1), shape=bhwc)
    x = vqmodel.decode(quant_z)
    return x


def load_numpy_image(im_path, width, height):
    im = Image.open(im_path).resize((width, height), resample=Image.LANCZOS)
    im = np.array(im)
    return im


def preprocess(rgbs, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]):
    rgbs = torch.from_numpy(np.stack(rgbs))
    rgbs = rgbs.float().permute(0, 3, 1, 2) / 255.0

    mean = torch.as_tensor(mean, dtype=rgbs.dtype, device=rgbs.device)
    std = torch.as_tensor(std, dtype=rgbs.dtype, device=rgbs.device)

    rgbs.sub_(mean[None, :, None, None]).div_(std[None, :, None, None])
    return rgbs


def load_vqmodel(fpath, device):
    config = OmegaConf.load(fpath)
    model = instantiate_from_config(config.model)
    model = model.to(device)
    return model