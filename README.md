# StableRep

<p align="center">
  <img src="figures/model.png" width="350">
</p>

This repo contains the PyTorch implementation of 
the [StableRep paper](https://arxiv.org/abs/2306.00984):

```bib
@inproceedings{tian2023stablerep,
  title={StableRep: Synthetic Images from Text-to-Image Models Make Strong Visual Representation Learners},
  author={Tian, Yonglong and Fan, Lijie and Isola, Phillip and Chang, Huiwen and Krishnan, Dilip},
  booktitle={NeurIPS},
  year={2023}
}
```

This is not an officially supported Google product.

## Prerequisites

- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN

We provide a conda `environment.yml` file listing the packages required. You can create
a conda environment via:
```commandline
conda env create -f environment.yml
```
If this does not work for you, try to switch to `environment_overcomplete.yml`.

## Data Generation

The instruction for data synthesis is described 
under `data_generation` folder.

## Training

You will need a csv file that specifies the paths to all images, you can either download 
from [here](https://www.dropbox.com/scl/fo/pk3yj5w7fa9l7a8a9ywoo/h?rlkey=b0uu7n96sahvchkiqf7eu9bj6&dl=0)
or generate by yourself for your own dataset.

By default, we use distributed multi-node training. Typical example of command on 
the first node is like:
```commandline
torchrun --nproc_per_node=8 --nnodes=4 \
  --node_rank=0 --master_addr="your host" --master_port=12345 \
  main_stablerep.py \
    --model base \
    --batch_size 43 \
    --epochs 15 --warmup_epochs 0.5 \
    --blr 2.0e-4 --weight_decay 0.1 --beta1 0.9 --beta2 0.98 \
    --num_workers 14 \
    --output_dir /path/to/output_model \
    --log_dir /path/to/output_log \
    --csv_path /path/to/csv_file \
    --folder_list /data/path1 /data/path2 /data/path3 ... \
    --n_img 6 --downsample --downsample_prob 0.05 --down_res 64 128
```
On other nodes, change `--node_rank` accordingly.

You can turn on the language tower to add extra image-text contrastive learning loss
(resultant model called StableRep++), simply by adding flag `--add_language`.

To reproduce the `1x`, `2x`, or `3x` schedules in the paper, simply set the `--epochs` and `--warmup_epochs`
as below:

|         |  schedule   | --epochs | --warmup_epochs | 
|---------|:-----------:|:--------:|:---------------:|
| cc12m   | 1x (35 ep)  |    15    |       0.5       |
|         | 2x (70 ep)  |    31    |       1.0       |
|         | 3x (105 ep) |    46    |       1.5       | 
| redcaps | 1x (35 ep)  |    13    |      0.45       | 
|         | 2x (70 ep)  |    27    |       0.9       | 
|         | 3x (105 ep) |    40    |      1.35       |

## Evaluation

For ImageNet linear probing, run the following command on a single node:
```commandline
torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr="your_host" --master_port=12345 \
  main_linear.py --model base --data /path/to/imagenet \
  --pretrained /path/to/pre-trained/epoch_last.pth \
  --output-dir /path/to/linear_save \
  --log-dir /path/to/tensorboard_folder
```
You can simply append `--use_bn` to turn on the extra BatchNorm (without affine transform)
layer for the linear classifiers.


## Results and Pre-trained Models

Pre-trained checkpoints can be found in this [link](https://www.dropbox.com/sh/1i1oilryhywgo9w/AAA6OTKU9uMmaK43Zwk6bu2Ma?dl=0).

(1) cc12m and redcaps: we generate 10 images for each caption, and use 6 out of 10 for each batch.

|         | schedule | downsample? | add text? | ImageNet Acc w/o BN | ImageNet Acc w/ BN | 
|:-------:|:--------:|:-----------:|:---------:|:-------------------:|:------------------:|
|  cc12m  |    1x    |             |           |        72.8         |        73.7        |
|         |    1x    |     Yes     |           |        71.4         |        73.5        |
|         |    3x    |             |           |        75.7         |        75.6        |
|         |    3x    |     Yes     |           |        75.4         |        75.2        |
|         |    1x    |             |    Yes    |        74.4         |        74.3        |
| redcaps |    1x    |             |           |        73.7         |        74.6        |
|         |    1x    |     Yes     |           |        73.4         |        74.6        |
|         |    3x    |             |           |        76.7         |        76.6        |
|         |    3x    |     Yes     |           |        76.4         |        76.4        |
|         |    1x    |             |    Yes    |        75.4         |        75.4        |

You can do downsample and adding text at the same time, but we do not have check points here.

(2) laion-50m subset: we generate 2 images per caption, and use both of them in each batch.

|             | scale | ImageNet Acc w/o BN | ImageNet Acc w/ BN | 
|-------------|:-----:|:-------------------:|:------------------:|
| StableRep++ |  1M   |        63.2         |        63.0        |
|             |  3M   |        69.6         |        69.5        |
|             |  10M  |        73.5         |        73.4        |
|             |  20M  |        73.9         |        73.8        |
|             |  50M  |        74.1         |        74.0        |
| CLIP (real images)       |  3M   |        60.6         |        60.6        |
|             |  10M  |        69.9         |        69.7        |
|             |  20M  |        71.5         |        71.5        |
|             |  50M  |        72.9         |        73.0        |

## Contact

For any questions related to the paper, please contact:

Yonglong Tian (yonglong@google.com)   
Lijie Fan (lijiefan@mit.edu)
