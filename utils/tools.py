import tempfile
import os
import tensorflow as tf
import numpy as np
from numpy.lib.stride_tricks import as_strided
import scipy.io as scio


def calc_SNR(recon, label):
    recon = np.array(recon).flatten()
    label = np.array(label).flatten()
    err = np.linalg.norm(label - recon) ** 2
    snr = 10 * np.log10(np.linalg.norm(label) ** 2 / err)

    return snr


def calc_PSNR(recon, label):
    recon = np.array(recon).flatten()
    label = np.array(label).flatten()
    err = np.linalg.norm(label - recon) ** 2
    max_label = np.max(np.abs(label))
    N = np.prod(recon.shape)
    psnr = 10 * np.log10(N * max_label ** 2 / err)

    return psnr


def mse(recon, label):
    if recon.dtype == tf.complex64:
        residual_cplx = recon - label
        residual = tf.stack([tf.math.real(residual_cplx), tf.math.imag(residual_cplx)], axis=-1)
        mse = tf.reduce_mean(residual ** 2)
    else:
        residual = recon - label
        mse = tf.reduce_mean(residual ** 2)
    return mse


def fft2c_mri(x):
    # nb nx ny nt
    X = tf.signal.ifftshift(x, 2)
    X = tf.transpose(X, perm=[0, 1, 3, 2])  # permute to make nx dimension the last one.
    X = tf.signal.fft(X)
    X = tf.transpose(X, perm=[0, 1, 3, 2])  # permute back to original order.
    nb, nt, nx, ny = np.float32(x.shape)
    nx = tf.constant(np.complex64(nx + 0j))
    ny = tf.constant(np.complex64(ny + 0j))
    X = tf.signal.fftshift(X, 2) / tf.sqrt(nx)

    X = tf.signal.ifftshift(X, 3)
    X = tf.signal.fft(X)
    X = tf.signal.fftshift(X, 3) / tf.sqrt(ny)

    return X


def ifft2c_mri(X):
    # nb nx ny nt
    x = tf.signal.ifftshift(X, 2)
    x = tf.transpose(x, perm=[0, 1, 3, 2])  # permute a to make nx dimension the last one.
    x = tf.signal.ifft(x)
    x = tf.transpose(x, perm=[0, 1, 3, 2])  # permute back to original order.
    nb, nt, nx, ny = np.float32(X.shape)
    nx = tf.constant(np.complex64(nx + 0j))
    ny = tf.constant(np.complex64(ny + 0j))
    x = tf.signal.fftshift(x, 2) * tf.sqrt(nx)

    x = tf.signal.ifftshift(x, 3)
    x = tf.signal.ifft(x)
    x = tf.signal.fftshift(x, 3) * tf.sqrt(ny)

    return x


def sos(x):
    # x: nb, ncoil, nt, nx, ny; complex64
    x = tf.math.reduce_sum(tf.abs(x ** 2), axis=1)
    x = x ** (1.0 / 2)
    return x

