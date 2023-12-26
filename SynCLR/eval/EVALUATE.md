# Evaluation of SynCLR

This is not an officially supported Google product.

## ImageNet Linear Probing

You should be able to do it by using the code in the [StableRep](../../StableRep) folder. Try to turn on `--use_bn`. 

|          | Top-1 Acc. |               Log                | 
|----------|:----------:|:--------------------------------:|
| ViT-B/16 |    80.7    | [log](logs/vit_b_linear_log.txt) |
| ViT-L/14 |    83.0    | [log](logs/vit_l_linear_log.txt) |

## ImageNet fine-tuning

You can fine-tune the pre-trained weights on ImageNet classification via the following command:

```commandline
export blr=5e-5
export layer_decay=0.65
export model_ema_decay=0.9999
torchrun --nproc_per_node=8 \
  --nnodes=1 --node_rank=0 --master_addr=localhost \
  --master_port=12345 \
  main_finetune.py \
    --batch_size 128 \
    --seed 0 \
    --model vit_base_patch16 \
    --epochs 100 \
    --warmup_epochs 20 \
    --blr ${blr} --layer_decay ${layer_decay} \
    --weight_decay 0.05 --drop_path 0.1 --reprob 0.25 --mixup 0.8 --cutmix 1.0 \
    --dist_eval \
    --data_path /dev/shm/imagenet \
    --nb_classes 1000 \
    --finetune /path/to/pre-trained/model.pth \
    --output_dir /path/to/output \
    --log_dir /path/to/log \
    --num_workers 10 --cls_token --crop_pct 0.95 \
    --model_ema --model_ema_eval --model_ema_decay ${model_ema_decay}
```

For ViT-L/14 fine-tuning, please follow the hyper-parameters listed in Table 14 of the paper.

You will get results similar to:

|          | Top-1 Acc. |                Log                 | 
|----------|:----------:|:----------------------------------:|
| ViT-B/16 |    85.8    | [log](logs/vit_b_imagenet_log.txt) |
| ViT-L/14 |    87.9    | [log](logs/vit_l_imagenet_log.txt) |


## Fine-grained evaluation

The command for evaluating the representations on fine-grained classification is given as:
```commandline
export CUDA_VISIBLE_DEVICES=0
export dataset='pets'
export iter=500
export model="/path/to/model.pth"
python main_downstream_linear.py -b 32 --model $model --dataset $dataset -a  vit_base_patch16 --max-iter $iter
```

## Semantic segmentation on ADE20k

We used `mmsegmentation` to conduct this experiment, and the configurations are listed under the `seg` folder.

|          | mIoU  |                                                             model                                                             |                                                          config                                                           | 
|----------|:-----:|:-----------------------------------------------------------------------------------------------------------------------------:|:-------------------------------------------------------------------------------------------------------------------------:|
| ViT-B/16 | 54.25 |       [model](https://www.dropbox.com/scl/fi/p0i594hzyg7ihumjoh41r/iter_48000.pth?rlkey=jmdv0z1ka8w4gt3gu3few69i1&dl=0)       |  [config](https://www.dropbox.com/scl/fi/uhf2zifh8mmkzh2rep0mu/mae-finetune-60k.py?rlkey=cyxgcw4wvnmxcy890drddaqop&dl=0)  |
| ViT-L/14 | 57.65 |      [model](https://www.dropbox.com/scl/fi/xzj13nfhmjigekh0cqyy1/iter_140000.pth?rlkey=6rxu2vtjfpgy64qo8qvwyhp8m&dl=0)       | [config](https://www.dropbox.com/scl/fi/f7n35nxv32tbjv9p0j1u5/mae-finetune-large.py?rlkey=lk72he5016x06ryhrsgcxikk0&dl=0) |