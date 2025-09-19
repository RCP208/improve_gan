# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Frechet Inception Distance (FID) from the paper
"GANs trained by a two time-scale update rule converge to a local Nash
equilibrium". Matches the original implementation by Heusel et al. at
https://github.com/bioinf-jku/TTUR/blob/master/fid.py"""

import numpy as np
import scipy.linalg
from . import metric_utils
import dill

#----------------------------------------------------------------------------

def compute_fid(opts, max_real, num_gen, swav=False, sfid=False):
    # Direct TorchScript translation of http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz
    detector_url = 'https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl'
    detector_kwargs = dict(return_features=True) # Return raw features before the softmax layer.

    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=0, capture_mean_cov=True, max_items=max_real, swav=swav, sfid=sfid).get_mean_cov()

    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_generator(
        opts=opts, detector_url=detector_url, detector_kwargs=detector_kwargs,
        rel_lo=0, rel_hi=1, capture_mean_cov=True, max_items=num_gen, swav=swav, sfid=sfid).get_mean_cov()

    if opts.rank != 0:
        return float('nan')
    print('prepare done')
    m = np.square(mu_gen - mu_real).sum()
    n = np.dot(sigma_gen, sigma_real)
    # cov = dict(cov=n)
    # with open('./cov_matrix.pth', 'wb') as f:
    #     dill.dump(cov, f)
    # print('compute eigvalue')
    # eigenvalue = np.linalg.eigvals(n)
    # if np.all(eigenvalue > 0):
    #     print('matrix is positive define')
    # else:
    #     print('matrix is non positive define')
    # print('start compute sqrtm')
    # U, S, Vt = np.linalg.svd(n)
    # s = U @ np.diag( S ** 0.5) @ Vt
    # print('compute sqrtm')
    s, _ = scipy.linalg.sqrtm(n, disp=False) # pylint: disable=no-member
    # if not np.isfinite(s).all():
    #     eps = 1e-6
    #     msg = ('fid calculation produces singular product; '
    #            'adding %s to diagonal of cov estimates') % eps
    #     print(msg)
    #     offset = np.eye(sigma_gen.shape[0]) * eps
    #     s = scipy.linalg.sqrtm((sigma_gen + offset).dot(sigma_real + offset))

    # Numerical error might give slight imaginary component
    # if np.iscomplexobj(s):
    #     if not np.allclose(np.diagonal(s).imag, 0, atol=1e-3):
    #         m = np.max(np.abs(s.imag))
    #         raise ValueError('Imaginary component {}'.format(m))
    #     s = s.real
    fid = np.real(m + np.trace(sigma_gen + sigma_real - s * 2))
    return float(fid)

#----------------------------------------------------------------------------
