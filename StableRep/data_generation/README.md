## data generation using Stable Diffusion

### Installation

1. Install Stable Diffusion Version 2 from [here](https://github.com/Stability-AI/stablediffusion). 
Recommend turning on XFormer.


2. Copy the `v1-inference.yaml` from [here](https://github.com/CompVis/stable-diffusion/tree/main/configs/stable-diffusion),
and put it under the `configs/stable-diffusion` of your Stable Diffusion v2 repo.


3. Download the `v1-5` weights from [this link](https://huggingface.co/runwayml/stable-diffusion-v1-5).


4. Copy the `txt2img.py` file from this folder to `scripts/` in your SD v2 repo.


### Captions

All captions used in this paper (cc3m, cc12m, redcaps, laion) are from 
public datasets, which can also be find from 
[here](https://www.dropbox.com/scl/fo/pk3yj5w7fa9l7a8a9ywoo/h?rlkey=b0uu7n96sahvchkiqf7eu9bj6&dl=0).


### Running

This code allows you to synthesize images with multiple GPUs on multiple machines. 
Let's say you have 8 machines and each of them contains 8 V-100 GPUs, then you can run the generation by

```commandline
node_idx=0  # change this number for each machine
n_gpus=8
n_nodes=8
for ((i=0;i<$n_gpus;i++)); do
    export CUDA_VISIBLE_DEVICES=$i
    python scripts/txt2img.py --outdir /path/to/output/ --seed 1 --scale 2.0 --batch_size 6 --split --from-file /path/to/prompt.txt --ckpt /path/to/ckpt.pt --gpu_idx $i --n_gpus $n_gpus --node_idx $node_idx --n_nodes $n_nodes &
done
```

This will generate one image for each caption. Then you can change the `--seed` to different values accordingly and 
generate more images for each caption. Our StableRep method needs multiple images
per caption, so make sure to change `--seed` to guarantee images are different within 
each caption.

If your machines happen to be ending with continuous numbers (e.g., 1,2,3,...), you can automatically
configure the `node_idx` instead of manual setup by something like:

```commandline
export node_base=1
export node_num=8
export MACHINE_NUM=$(hostname | sed 's/[^0-9]*//g')
export node_idx_abs=$(($MACHINE_NUM - $node_base))
export node_idx=$(($node_idx_abs % $node_num))
```
