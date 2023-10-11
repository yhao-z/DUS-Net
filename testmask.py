from dataset_tfrecord import get_dataset
from mask_generator import generate_mask
import numpy as np

datadir = '../data/'
mode = 'test' # val or test
masktype = 'vds_0.1'

dataset_test = get_dataset(mode, datadir, 1, shuffle=False)
MASK = []
for step, sample in enumerate(dataset_test):
    k0, label = sample

    # generate under-sampling mask (random)
    nb, nt, nx, ny = k0.get_shape()
    mask = generate_mask([nx, ny, nt], float(masktype.split('_', 1)[1]), masktype.split('_', 1)[0])
    mask = np.transpose(mask, (2, 0, 1))
    mask = np.complex64(mask + 0j)
    MASK.append(mask)
np.savez(datadir+mode+'_'+masktype, *MASK)