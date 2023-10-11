import tensorflow as tf
from utils.tools import fft2c_mri, ifft2c_mri
import scipy.io as scio


class CONV_OP(tf.keras.layers.Layer):
    def __init__(self, n_f=32, ifactivate=False):
        super(CONV_OP, self).__init__()
        self.mylayers = []
        self.mylayers.append(tf.keras.layers.Conv3D(n_f, 3, strides=1, padding='same', use_bias=False))
        if ifactivate == True:
            self.mylayers.append(tf.keras.layers.ReLU())
        self.seq = tf.keras.Sequential(self.mylayers)

    def call(self, input):
        res = self.seq(input)
        return res

class DUS_Net_s(tf.keras.Model):
    def __init__(self, niter, channels):
        super(DUS_Net_s, self).__init__(name='DUS_Net_s')
        self.niter = niter
        self.channels = channels
        self.celllist = []  

    def build(self, input_shape):
        for i in range(self.niter - 1):
            self.celllist.append(DUS_s_Cell(input_shape, i, self.channels))
        self.celllist.append(DUS_s_Cell(input_shape, self.niter - 1, self.channels, is_last=True))

    def call(self, d, mask):
        """
        d: undersampled k-space
        csm: coil sensitivity map
        mask: sampling mask
        label: to test the SNRs, when test passed, it can be delete
        """
        # nb, nc, nt, nx, ny = d.shape
        x_rec = ifft2c_mri(d)
        # scio.savemat('./inner_study/d.mat', {'x': x_rec.numpy()})
        x_sym = tf.zeros_like(x_rec)
        Z = tf.zeros_like(x_rec)
        L = tf.zeros_like(x_rec)
        data = [x_rec, x_sym, Z, L, d, mask]

        X_SYM = []
        for i in range(self.niter):
            data = self.celllist[i](data)
            # scio.savemat('./inner_study/d%d.mat' % i, {'x': data[0].numpy(), 'Z': data[3].numpy()})
            x_sym = data[1]
            # scio.savemat('./inner_study/sym%d.mat' % i, {'sym': x_sym.numpy()})
            X_SYM.append(x_sym)

        x_rec = data[0]

        return x_rec, X_SYM


class DUS_s_Cell(tf.keras.layers.Layer):
    def __init__(self, input_shape, i, channels, is_last=False):
        super(DUS_s_Cell, self).__init__()
        self.nb, self.nt, self.nx, self.ny = input_shape

        self.thres_coef = tf.Variable(tf.constant(-2, dtype=tf.float32), trainable=True, name='thres_coef %d' % i)
        self.mu = tf.Variable(tf.constant(0.1, dtype=tf.float32), trainable=True, name='mu %d' % i)
        self.eta = tf.Variable(tf.constant(1, dtype=tf.float32), trainable=(not is_last), name='eta %d' % i)

        self.conv_1 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_2 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_3 = CONV_OP(n_f=channels, ifactivate=False)

        self.conv_4 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_5 = CONV_OP(n_f=channels, ifactivate=True)
        self.conv_6 = CONV_OP(n_f=2, ifactivate=False)

    def call(self, data, **kwargs):
        x_rec, x_sym, Z, L, d, mask = data

        Z, x_sym = self.prox_step(x_rec + L)
        x_rec = self.admm_x_step(L, Z, d, mask)
        L = self.admm_L_step(L, x_rec, Z)

        data[0] = x_rec
        data[1] = x_sym
        data[2] = Z
        data[3] = L

        return data

    def prox_step(self, x):
        [batch, Nt, Nx, Ny] = x.get_shape()
        x_in = tf.stack([tf.math.real(x), tf.math.imag(x)], axis=-1)
        x_sym = 0
        x_1 = self.conv_1(x_in)
        x_2 = self.conv_2(x_1)
        x_3 = self.conv_3(x_2)

        x_3 = tf.math.multiply(tf.math.sign(x_3), tf.nn.relu(tf.abs(x_3) - tf.nn.relu(self.thres_coef)))

        x_1_sym = self.conv_4(x_3)
        x_1_sym = self.conv_5(x_1_sym)
        x_1_sym = self.conv_6(x_1_sym)
        # x_1_sym = self.conv_7(x_1_sym)
        x_sym = x_1_sym - x_in  
        
        x_4 = self.conv_4(x_3)
        x_5 = self.conv_5(x_4)
        x_6 = self.conv_6(x_5)
        # x_7 = self.conv_7(x_6)

        Z = x_6 + x_in
        Z = tf.complex(Z[:, :, :, :, 0], Z[:, :, :, :, 1])

        return Z, x_sym

    def admm_x_step(self, L, Z, d, mask):
        temp = Z - L
        k_rec = fft2c_mri(temp)  # tf.cast(tf.nn.relu(self.mu), tf.complex64)
        k_rec = tf.math.scalar_mul(tf.cast(tf.nn.relu(self.mu), tf.complex64), d) + k_rec
        k_rec = tf.math.divide_no_nan(k_rec, tf.math.scalar_mul(tf.cast(tf.nn.relu(self.mu), tf.complex64), mask) + 1)
        x_rec = ifft2c_mri(k_rec)
        return x_rec

    def admm_L_step(self, L, x_rec, Z):
        eta = tf.cast(tf.nn.relu(self.eta), tf.complex64)
        return L + tf.math.scalar_mul(eta, x_rec - Z)



