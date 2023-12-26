# Pytorch reference training code


## Disclaimer
The models released in this repo were not from the code here. Instead, 
they were trained using Jax inside 
Google's computation framework. To facilitate reproduction of our work 
outside Google, we try to provide a Pytorch reference code here. We haven't
checked the performance of using this pytorch reference code on the dataset
synthesized from the SynCaps-150M used in our paper, becasue of various reasons.


## Data

This code assumes the synthetic data was stored in the similar format as that
in [StableRep](../../StableRep). Check for more details in that folder.


## Training
An example of training command on the first node (of multi-node training) is:
```commandline
export n_img=4
export batchsize=64
export n_epoch=15
export n_warmup_epoch=0.5
torchrun --nproc_per_node=8 --nnodes=4 \
  --node_rank=${node_rank} --master_addr=${exp_host} --master_port=12345 \
  main_mask_synclr.py \
    --model base \
    --batch_size ${batchsize} \
    --epochs ${n_epoch} --warmup_epochs ${n_warmup_epoch} \
    --blr 1.5e-4 --weight_decay 0.1 --weight_decay_end 0.1 --beta1 0.9 --beta2 0.98 \
    --num_workers 12 \
    --output_dir ./output/ \
    --log_dir ./log/ \
    --csv_path /path/to/csv_file/ \
    --folder_list /folder1 /folder2 /folder3 /folder4 \
    --n_img ${n_img} --local_crops_number 4 \
    --moco-m 0.994 --moco-m-epochs ${n_epoch} --moco-m-cos \
    --contrast_loss_weight 1.0 --ibot_loss_weight 1.0 \
    --mask_probability 0.5 --mask_style ibot --mask_shape rand --mask_ratio 0.5 --mask_first_n \
    --dino_output_dim 8192 --warmup_teacher_temp_epochs 1.0
```
