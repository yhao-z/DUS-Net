# DUS-Net ( IEEE ICIP, 2023 )

The official TensorFlow implementation of **Deep Unrolling Shrinkage Network for Dynamic MR Imaging** ([conference](https://ieeexplore.ieee.org/abstract/document/10223077)|[arXiv](https://arxiv.org/abs/2307.09818))

Deep unrolling networks that utilize sparsity priors have achieved great success in dynamic magnetic resonance (MR) imaging. The convolutional neural network (CNN) is usually utilized to extract the transformed domain, and then the soft thresholding (ST) operator is applied to the CNN-transformed data to enforce the sparsity priors. However, the ST operator is usually constrained to be the same across all channels of the CNN-transformed data. In this paper, we propose a novel operator, called soft thresholding with channel attention (AST), that learns the threshold for each channel. In particular, we put forward a novel deep unrolling shrinkage network (DUS-Net) by unrolling the alternating direction method of multipliers (ADMM) for optimizing the transformed $l_1$ norm dynamic MR reconstruction model. Experimental results on an open-access dynamic cine MR dataset demonstrate that the proposed DUS-Net outperforms the state-of-the-art methods.

![net](https://yhao-img-bed.obs.cn-north-4.myhuaweicloud.com/202310110952647.png)

## 1. Getting Started

### Environment Configuration

- we recommend to use docker

  ```shell
  # pull the docker images
  docker pull yhaoz/tf:2.9.0-bart
  # then you can create a container to run the code, see docker documents for more details
  ```

- if you don't have docker, you can still configure it via installing the requirements by yourself

  ```shell
  pip install -r requirements.txt # tensorflow is gpu version
  ```

Note that, we only run the code in NVIDIA GPU. In our implementation, the code can run normally in both Linux & Windows system.

### Dataset preparation

You can download the dataset via [my OneDrive](https://stuhiteducn-my.sharepoint.com/:f:/g/personal/yhao-zhang_stu_hit_edu_cn/Ev1ZhrDUVU1EmJHg81y1-eYBdMRRbzb1SpXxQJtodMGsfg?e=NfFFXI). You may need to download the following files and put them in `./data` file folder.

```shell
# dataset for training, validation, test.
ocmr_train.tfrecord
ocmr_val.tfrecord
ocmr_test.tfrecord

# undersampling mask for validation and test. Only the radial with 16 lines are provided here.
# if you're interested in generating other masks, see testmask.py for more details.
val_radial_16.npz
test_radial_16.npz
```

The dataset pre-processing and creating code can be found in [yhao-z/ocmr-preproc-tf](https://github.com/yhao-z/ocmr-preproc-tf). 

## 2. Run the code

### Test only

We provide the training weights of our `DUS-Net` and `DUS_Net_s` (named DUS_Net$^-$ in the paper) for radial-16 sampling cases. Note that the provided weights are only applicable in our provided dataset with radial-16 sampling. **If you are using other different configuration, retraining from scratch may be needed.**

```shell
# test the DUS_Net
python test.py
# test the DUS_Net_s
python test.py --ModelName 'DUS_Net_s' --weight './weights/DUS_Net_s-radial_16/weight-best'
```

### Training

```shell
# Please refer to main.py for more configurations.
python main.py
```

## 3. Citation

If you find this work useful for your research, please cite:

```
@inproceedings{zhang2023deep,
  title={Deep Unrolling Shrinkage Network for Dynamic MR Imaging},
  author={Zhang, Yinghao and Li, Xiaodi and Li, Weihang and Hu, Yue},
  booktitle={2023 IEEE International Conference on Image Processing (ICIP)},
  pages={1145--1149},
  year={2023},
  organization={IEEE}
}
```

