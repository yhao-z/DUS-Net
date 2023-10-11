import argparse
import os
import sys
import tensorflow as tf
from Solver import Solver



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, default='0', help='GPU No.')
    
    parser.add_argument('--datadir', type=str, default='../data/')
    
    parser.add_argument('--start_epoch', type=int, default=1, help='start epoch, begin with 1')
    parser.add_argument('--end_epoch', type=int, default=50, help='end epoch or number of epochs')

    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='initial learning rate')

    parser.add_argument('--channels', type=int, default=16)
    parser.add_argument('--factor', type=float, default=0., help='hyperparameter for inverse constraint of two CNNs in each iteration')
    parser.add_argument('--niter', type=int, default=15, help='number of network iterations')
    parser.add_argument('--masktype', type=str, default='radial_16')

    parser.add_argument('--ModelName', type=str, default='DUS_Net', help='DUS_Net, DUS_Net_s')

    parser.add_argument('--weight', type=str, default='./weights/DUS_Net-radial_16/weight-best')

    args = parser.parse_args()
    
    # GPU setup
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    GPUs = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(GPUs[0], True)
    
    print(args)
    Solver(args).test()
