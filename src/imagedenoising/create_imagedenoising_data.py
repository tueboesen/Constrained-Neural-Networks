import math
import time

import numpy as np
import torch
import torch.nn.functional as F

def Ax(x):
    x1 = div_cc(x, mode=1)
    x2 = div_cc(x1, mode=0) + 1e-3 * x
    return x2

def div_free(UVo):
    # Generate a div free projection
    # min 0.5*\|UV - UVo\|^2 s.t div UV = 0
    # Gives the KKT system
    #  1) UV - UVo - div^T x = 0
    #  2) div UV = 0
    #  UV = UVo + div^T x => div*divT x = -divUVo
    # Use CG to solve the system
    # Ax2 = lambda x: div_cc(div_cc(x, mode=1), mode=0) + 1e-3 * x

    b = div_cc(UVo)
    x = torch.zeros_like(b)
    r = b
    p = r
    # Conjugate gradient
    for i in range(5000):
        Ap = Ax(p)
        alpha = (r * r).mean() / (p * Ap).mean()
        x = x + alpha * p
        r = b - Ax(x)
        p = r
        if r.norm() / b.norm() < 1e-7:
            UV = UVo - div_cc(x, mode=1)
            return UV
        if (i%100) == 0:
            print('%3d   %3.2e' % (i, r.norm() / b.norm()))
    UV = UVo - div_cc(x, mode=1)
    return UV


def div_cc(UV, mode=0):
    # Do it with convolutions
    Dx = torch.tensor([[-1.0, 1.0], [-1, 1]],device=UV.device).reshape(1, 1, 2, 2)
    Dy = torch.tensor([[-1.0, -1.0], [1, 1]],device=UV.device).reshape(1, 1, 2, 2)
    D = torch.cat((Dx, Dy), dim=0)

    # So the first of the two images is convolved with [-1 1]
    #                                                  [-1 1]
    # While the second image is convolved with [-1 -1]
    #                                          [ 1  1]
    if mode == 0:  # Compute the div
        dUV = F.conv2d(UV, D, groups=2)
        dUV = dUV.sum(dim=1, keepdim=True)
    else:  # Compute the adjoint
        UV = torch.cat((UV, UV), dim=1)
        dUV = F.conv_transpose2d(UV, D, groups=2)
    return dUV


def create_and_save_imagedenoising_data(nsamples, npixels, file):
    UV = torch.randn(nsamples, 2, npixels, npixels)

    # Q = div_cc(UV)
    # div = F.mse_loss(Q, torch.zeros_like(Q))
    # plot_img(UV,title=div.item())
    # Make it non-trivial by correlations
    # This is actually not needed, but is merely here to make the images have more largescale texture and look more interesting, right?
    K = torch.randn(2, 2, 5, 5)
    K = K - K.sum(dim=[2, 3], keepdim=True)
    for i in range(10):
        UV = F.conv2d(UV, K, padding=2)
        UV = UV / UV.abs().mean()

    UVdvf = div_free(UV)

    Q = div_cc(UVdvf)
    div = F.mse_loss(Q, torch.zeros_like(Q))
    # plot_img(UVdvf,title=div.item())
    print('Div U = %3.2e' % div)
    np.savez(file, images=UVdvf, divergence=div)
    return


if __name__ == '__main__':
    nsamples = 500
    npixels = 64
    file = '../../data/imagedenoising/images.npz'
    create_and_save_imagedenoising_data(nsamples, npixels, file)
# models.animate_pendulum()
