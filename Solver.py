# import
import argparse
import os
import sys
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
import scipy.io as scio
# inner import
from model import DUS_Net
from models.DUS_Net_s import DUS_Net_s
from utils.dataset_tfrecord import get_dataset
from utils.tools import mse, calc_SNR, calc_PSNR, ifft2c_mri
from utils.mask_generator import generate_mask


class Solver(object):
    def __init__(self, args):
        self.datadir = args.datadir
        
        self.start_epoch = args.start_epoch
        self.end_epoch = args.end_epoch
        
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.channels = args.channels
        self.factor = args.factor
        self.niter = args.niter
        self.masktype = args.masktype
        self.ModelName = args.ModelName
        self.weight = args.weight
        # specify network
        if self.ModelName == 'DUS_Net':
            self.net = DUS_Net(self.niter, self.channels)
        elif self.ModelName == 'DUS_Net_s':
            self.net = DUS_Net_s(self.niter, self.channels)
        self.param_num = 0 # initialize it 0, later calc it
        
        
    def train(self):
        # give the log dir and the model dir
        name_seq = [str(self.ModelName), str(self.masktype), str(self.niter), str(self.channels), str(self.factor), str(self.batch_size), str(self.learning_rate)]
        model_id = '-'.join([name_seq[i] for i in [0,1]]) # it can be chosen flexiably
        TIMESTAMP = "{0:%Y%m%dT%H%M%S}".format(datetime.now())
        # log
        train_logdir = os.path.join('./logs/train/', model_id + '-' + TIMESTAMP)
        val_logdir = os.path.join('./logs/val/', model_id + '-' + TIMESTAMP)
        train_writer = tf.summary.create_file_writer(train_logdir)
        val_writer = tf.summary.create_file_writer(val_logdir)
        
        # model
        os.makedirs('./weights/') if not os.path.exists('./weights/') else None
        weightdir = os.path.join('./weights/', model_id + '-' + TIMESTAMP)
        
        # prepare dataset
        dataset_train = get_dataset('train', self.datadir, self.batch_size, shuffle=True)
        dataset_val = get_dataset('val', self.datadir, 1, shuffle=False)
        tf.print('dataset loaded.')

        # load pre-weight
        if self.weight is not None:
            tf.print('load weights.')
            self.net.load_weights(self.weight)
        tf.print('network initialized.')

        # define lr and optimizer
        learning_rate = self.learning_rate
        learning_rate_decay = 0.95
        learning_rate = learning_rate * learning_rate_decay ** (self.start_epoch - 1)
        optimizer = tf.optimizers.Adam(learning_rate)

        # Iterate over epochs.
        total_step = 0
        loss = 0
        
        # calculate the base validate PSNR
        # start epoch equals to 1 means no pre-trained weights, so we calc psnr using the original undersampled data
        # else we calc psnr using the reconstructed data from net using the pre-trained weights
        if self.weight is not None:
            self.net.load_weights(os.path.split(self.weight)[0]+'/weight-best')
            val_psnr_best, _, _, _ = self.val(dataset_val, is_first=(self.start_epoch==1))
            self.net.load_weights(self.weight)
        else:
            val_psnr_best, _, _, _ = self.val(dataset_val, is_first=(self.start_epoch==1))
        print(20*'*')
        print('the best val psnr is /%.3f/' % val_psnr_best)
        print(20*'*')

        for epoch in range(self.start_epoch, self.end_epoch+1):
            for step, sample in enumerate(dataset_train):
                # forward
                t0 = time.time()
                k0 = None
                with tf.GradientTape() as tape:
                    k0, label = sample

                    if k0 is None:
                        continue
                    if k0.shape[0] < self.batch_size:
                        continue

                    # generate under-sampling mask (random)
                    nb, nt, nx, ny = k0.get_shape()
                    mask = generate_mask([nx, ny, nt], float(self.masktype.split('_', 1)[1]), self.masktype.split('_', 1)[0])
                    mask = np.transpose(mask, (2, 0, 1))
                    mask = tf.constant(np.complex64(mask + 0j))

                    # generate the undersampled data k0
                    k0 = k0 * mask

                    # feed the data
                    recon, X_SYM = self.net(k0, mask)
                    recon_abs = tf.abs(recon)
                    psnr = calc_PSNR(recon, label)

                    # compute loss
                    loss, mse, sym = loss_func(recon, label, X_SYM, factor=self.factor, sym = True)
                    
                # sum all the losses and avarage them to write the summary when epoch ends
                psnr_epoch = psnr_epoch + psnr if step != 0 else psnr
                loss_epoch = loss_epoch + loss.numpy() if step != 0 else loss.numpy()
                mse_epoch = mse_epoch + mse.numpy() if step != 0 else mse.numpy()
                sym_epoch = sym_epoch + sym.numpy() if step != 0 else sym.numpy()

                # backward
                grads = tape.gradient(loss, self.net.trainable_weights)
                optimizer.apply_gradients(zip(grads, self.net.trainable_weights)) 
                
                if self.param_num == 0:
                    self.param_num = np.sum([np.prod(v.get_shape()) for v in self.net.trainable_variables])

                # log output
                if step % 100 == 0: 
                    tf.print('Epoch', epoch, '/', self.end_epoch, 'Step', step, 'loss =', '%.3e' % loss.numpy(), 'PSNR =', '%.2f' % psnr, 'time',
                            '%.2f' % (time.time() - t0), 'lr = ', '%.4e' % learning_rate, 'param_num', self.param_num)
                total_step += 1
                
            # At the end of epoch, print one message
            tf.print('Epoch', epoch, '/', self.end_epoch, 'Step', step, 'loss =', '%.3e' % loss.numpy(), 'PSNR =', '%.2f' % psnr, 'time',
                    '%.2f' % (time.time() - t0), 'lr = ', '%.4e' % learning_rate, 'param_num', self.param_num)
            
            # record loss
            with train_writer.as_default():
                tf.summary.scalar('loss/loss', loss_epoch/(step+1), step=epoch)
                tf.summary.scalar('loss/mse', mse_epoch/(step+1), step=epoch)
                tf.summary.scalar('loss/sym', sym_epoch/(step+1), step=epoch)
                tf.summary.scalar('PSNR', psnr_epoch/(step+1), step=epoch)

            # learning rate decay for each epoch
            learning_rate = learning_rate * learning_rate_decay
            optimizer = tf.optimizers.Adam(learning_rate)
            
            # validate
            val_psnr, val_loss, val_mse, val_sym = self.val(dataset_val)
            with val_writer.as_default():
                tf.summary.scalar('loss/loss', val_loss, step=epoch)
                tf.summary.scalar('loss/mse', val_mse, step=epoch)
                tf.summary.scalar('loss/sym', val_sym, step=epoch)
                tf.summary.scalar('PSNR', val_psnr, step=epoch)
                
            # save model
            # if validate PSNR is better than the best PSNR, save the best model
            if val_psnr > val_psnr_best:
                self.net.save_weights(weightdir+'/weight-best')
                print(20*'*')
                print('epoch %d' % epoch, 'best PSNR = %.2f' % val_psnr)
                print(20*'*')
                val_psnr_best = val_psnr
            # save the latest epoch weights for continued training
            self.net.save_weights(weightdir+'/weight-latest')
            # every 10 epoches, we save the weights
            if epoch % 10 == 0 or epoch == 1:
                self.net.save_weights(weightdir+'/weight-'+str(epoch)) 

    
    
    def val(self, dataset_val, is_first = False):
        masks = np.load(self.datadir+'val_'+self.masktype+'.npz')
        for step, sample in enumerate(dataset_val):
            k0, label = sample

            # generate under-sampling mask (fix for val)S
            mask = masks[masks.files[step]]
            mask = tf.constant(np.complex64(mask + 0j))

            # generate the undersampled data k0
            k0 = k0 * mask

            # feed the data
            if is_first:
                recon = ifft2c_mri(k0)
                loss_all = 0
                mse_all = 0
                sym_all = 0
            else:
                recon, X_SYM = self.net(k0, mask)
                
                # compute loss
                loss, mse, sym = loss_func(recon, label, X_SYM, factor=self.factor, sym = True)
                
                loss_all = loss_all + loss.numpy() if step != 0 else loss.numpy()
                mse_all = mse_all + mse.numpy() if step != 0 else mse.numpy()
                sym_all = sym_all + sym.numpy() if step != 0 else sym.numpy()
                
            psnr_all = psnr_all + calc_PSNR(recon, label) if step != 0 else calc_PSNR(recon, label)

        val_loss = loss_all/(step+1)
        val_mse = mse_all/(step+1)
        val_sym = sym_all/(step+1)
        val_psnr = psnr_all/(step+1)
            
        return val_psnr, val_loss, val_mse, val_sym



    def test(self):
        dataset_test = get_dataset('test', self.datadir, 1, shuffle=False)
        tf.print('loading weights...')
        self.net.load_weights(self.weight)
        tf.print('net initialized, testing...')
        SNRs = []
        PSNRs = []
        MSEs = []
        SSIMs = []
        masks = np.load(self.datadir+'test_'+self.masktype+'.npz')
        for step, sample in enumerate(dataset_test):
            k0, label = sample

            # generate under-sampling mask (fix for test)
            mask = masks[masks.files[step]]
            mask = tf.constant(np.complex64(mask + 0j))

            # generate the undersampled data k0
            k0 = k0 * mask
            
            # feed the data
            t0 = time.time()
            recon, X_SYM = self.net(k0, mask)
            t = time.time() - t0
            
            if step == 8:
                scio.savemat(self.ModelName+'.mat', {'recon': recon.numpy()})
                scio.savemat('us.mat', {'us': ifft2c_mri(k0).numpy()})
            
            # calc the metrics
            SNR_ = calc_SNR(recon, label)
            PSNR_ = calc_PSNR(recon, label)
            MSE_ = mse(recon, label)
            SSIM_ = tf.image.ssim(tf.transpose(tf.abs(recon), [0, 2, 3, 1]), tf.transpose(tf.abs(label), [0, 2, 3, 1]), max_val=1.0)
            SNRs.append(SNR_)
            PSNRs.append(PSNR_)
            MSEs.append(MSE_)
            SSIMs.append(SSIM_)
            print('data %d --> SNR = \%.3f\, PSNR = \%.3f\, SSIM = \%.3f\, MSE = {%.3e}, t = %.2f' % (step, SNR_, PSNR_, SSIM_, MSE_, t))
            
        SNRs = np.array(SNRs)
        PSNRs = np.array(PSNRs)
        MSEs = np.array(MSEs)
        print('SNR = %.3f(%.3f), PSNR = %.3f(%.3f), SSIM = %.3f(%.3f), MSE = %.3e(%.3e)' % (np.mean(SNRs), np.std(SNRs), np.mean(PSNRs), np.std(PSNRs), np.mean(SSIMs), np.std(SSIMs), np.mean(MSEs), np.std(MSEs)))



def loss_func(y, y_, y_sym, factor=0.01, sym = True):
    pred = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
    label = tf.stack([tf.math.real(y_), tf.math.imag(y_)], axis=-1)

    cost = tf.reduce_mean(tf.math.square(pred - label))
    if sym:
        cost_sym = tf.reduce_mean(tf.square(y_sym))
    else:
        cost_sym = tf.reduce_mean(tf.square(tf.constant(y_sym, dtype=tf.float32)))
    loss = cost + factor * cost_sym
    return loss, cost, cost_sym


