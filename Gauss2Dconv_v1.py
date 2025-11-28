# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 10:38:59 2023

@author: tobjer
"""

import numpy as np
from scipy.fft import fft2, ifft2, fftshift, ifftshift

#%% Gauss2Dconv_v1


def Gauss2Dconv_v1(lon, lat, val, sigma):
    """
    Applies a 2D Gaussian filter to a grid by a convolution in the Fourier domain.

    Parameters:
    lon : 2D numpy array of longitude [deg]
    lat : 2D numpy array of latitude [deg]
    val : 2D numpy array of signal values to be filtered
    sigma : Scalar that determines the filter width [m]

    Returns:
    g_low : 2D numpy array of low-pass filtered values
    g_high : 2D numpy array of high-pass filtered values, i.e., val - g_low
    """

    # Conversion factor
    deg2rad = np.pi / 180
    R = 6371000  # Earth radius in meters

    # Get dimensions
    n, m = val.shape

    # Check that grid is consistent
    if lon.shape != (n, m) or lat.shape != (n, m):
        raise ValueError('Grid dimensions not consistent')

    # Derive grid increments
    dinc = lon[1, 2] - lon[1, 1]

    # Convert from degrees to meters
    sample_interval = dinc * deg2rad * R

    # Transform sigma from meters to degrees
    sigma = sigma / sample_interval

    # Account for NaN values
    val[np.isnan(val)] = 0

    # Form 2D Fourier Transform
    F = fft2(val)

    # Derive frequency
    N, M = F.shape
    k = np.fft.fftfreq(N).reshape(N, 1)
    l = np.fft.fftfreq(M).reshape(1, M)

    # Form variables with zero frequency at center
    F_center = fftshift(F)
    k_center = fftshift(k)
    l_center = fftshift(l)

    # Form Filter Kernel
    H_center = np.exp(-sigma**2 * (k_center**2 + l_center**2) / 2)

    # Apply Filter and Reconstruct Signal
    G_center = H_center * F_center
    G = ifftshift(G_center)

    # Form inverse FFT
    g_low = ifft2(G).real

    # Derive high-pass product
    g_high = val - g_low

    return g_low, g_high

