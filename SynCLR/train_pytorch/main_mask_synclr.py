# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import datetime
import json
import numpy as np
import os
import time
from pathlib import Path
import math
import sys
from typing import Iterable
from functools import partial

import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets
from torch.utils.tensorboard import SummaryWriter

from models.masking import BlockMaskingGenerator, RandomMaskingGenerator
from models.masking import Data2vecBlockMaskingGenerator, JepaBlockMaskingGenerator
from dataset.augmentations import MultiCropDataAugmentation
from dataset.data import SupconMultiCropDataset
from dataset.collate import collate_data_and_cast
from synclr_meta_arch import MetaArch

import util.lr_sched as lr_sched
import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('Synthetic Mask Training', add_help=False)

    # add dataset parameters
    parser.add_argument('--global_crops_scale', default=(0.32, 1.), type=float, nargs='+',
                        help="scale range for global crops, when disabling local crops, recommend"
                             "using a wider range of global crops (--global_crops_scale 0.14 1.)")
    parser.add_argument('--local_crops_scale', default=(0.08, 0.32), type=float, nargs='+',
                        help="cropping scales for local crops")
    parser.add_argument('--global_crops_number', default=1, type=int,
                        help="how many global crops per image")
    parser.add_argument('--local_crops_number', default=2, type=int,
                        help="how many local crops per image")
    parser.add_argument('--global_crops_size', '--img_size', default=224, type=int,
                        help="this should be equal to image size")
    parser.add_argument('--local_crops_size', default=96, type=int,
                        help="this should be divisible by patch size")
    parser.add_argument('--patch_size', default=16, type=int,
                        help="patch size for vit patch embedding")

    # add ImageNet dataset example
    parser.add_argument('--use_imagenet', action='store_true',
                        help='use imagenet dataset')
    parser.add_argument('--imagenet_root', default='/dev/shm/imagenet',
                        help='imagenet data path')

    parser.add_argument('--csv_path', default='./data/MAE/MAE_train.csv', type=str, help='csv file path')
    parser.add_argument('--folder_list', nargs='+', type=str, help='A list of folders')
    parser.add_argument('--n_img', type=int, default=1, help='number of images per caption sample,')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size per GPU.')

    # add masking parameter
    parser.add_argument('--mask_shape', default='rand', choices=['rand', 'block', 'jepa_block', 'd2v_block'], type=str,
                        help='mask shape, either rand or block')
    parser.add_argument('--mask_ratio', default=(0.1, 0.5), type=float, nargs='+',
                        help="mask ratio can be either a value or a range")
    parser.add_argument('--mask_ratio_adjust', default=0.07, type=float,
                        help="mask ratio adjust for d2v_block")
    parser.add_argument('--mask_probability', default=0., type=float,
                        help="how many samples with be applied with masking")
    parser.add_argument('--mask_first_n', action='store_true',
                        help="mask the first n sample to avoid shuffling. Needed for MAE-style encoder")
    parser.add_argument('--clone_batch', default=1, type=int,
                        help="how many times to clone the batch for masking (default: 1, not cloning)")
    parser.add_argument('--use_multi_pred', action='store_true',
                        help="use multiple predictions for masking")

    # add model parameters
    parser.add_argument('--model', default='base', choices=['small', 'base', 'large'], type=str,
                        help="the vit model to use: small, base, or large.")
    parser.add_argument('--mask_style', default='none', choices=['ibot', 'mae', 'none'], type=str,
                        help="specify encoder masking style, either mae or ibot, for now")
    parser.add_argument('--drop_path_rate', default=0.0, type=float,
                        help='Drop path rate for the student model (default: 0.0)')
    parser.add_argument('--contrast_mlp_dim', default=4096, type=int,
                        help='mlp dim for contrastive head')
    parser.add_argument('--contrast_output_dim', default=256, type=int,
                        help='output dim for contrastive head')
    parser.add_argument('--contrast_mlp_norm', default='ln', type=str, choices=['ln', 'bn'],
                        help='normalization layer for contrastive head')
    parser.add_argument('--contrast_temperature', default=0.1, type=float,
                        help='temperature for contrastive loss')
    parser.add_argument('--avoid_local_global_same', action='store_true',
                        help='avoid local and global crops to be from the same image for contrastive loss')
    parser.add_argument('--dino_mlp_dim', default=2048, type=int,
                        help='mlp dim for dino head')
    parser.add_argument('--dino_output_dim', default=65536, type=int,
                        help='output dim for dino head')
    parser.add_argument('--dino_bottleneck_dim', default=256, type=int,
                        help='bottleneck dim for dino head')
    parser.add_argument('--dino_n_layers', default=3, type=int,
                        help='number of layers for dino head')
    parser.add_argument('--do_feature_decoder', action='store_true',
                        help='whether to use feature decoder')
    parser.add_argument('--decoder_type', default='transformer', choices=['transformer', 'cnn'], type=str,
                        help='feature decoder type')
    parser.add_argument('--decoder_embed_dim', default=384, type=int,
                        help='feature decoder embed dim')
    parser.add_argument('--decoder_depth', default=6, type=int,
                        help='feature decoder depth')
    parser.add_argument('--decoder_num_heads', default=12, type=int,
                        help='how manay heads in the decoder (better set it equal to the num of heads in encoder')
    parser.add_argument('--decoder_kernel', default=3, type=int,
                        help='feature decoder kernel size, when using cnn decoder')
    parser.add_argument('--ffn_targets', action='store_true',
                        help='use ffn output as the target for regressing features')
    parser.add_argument('--return_layer_targets', action='store_true',
                        help='return layer targets for mae feature prediction')

    # add loss parameters
    parser.add_argument('--contrast_loss_weight', default=2.0, type=float,
                        help='weight for contrastive loss')
    parser.add_argument('--dino_loss_weight', default=0.0, type=float,
                        help='weight for dino prototype loss (image level)')
    parser.add_argument('--ibot_loss_weight', default=0.0, type=float,
                        help='weight for ibot prototype loss (patch level)')
    parser.add_argument('--mae_loss_weight', default=0.0, type=float,
                        help='weight for mae feature/pixel loss (patch level)')
    parser.add_argument('--centering', default='sinkhorn_knopp', type=str, choices=['sinkhorn_knopp', 'centering'],
                        help='centering method for dino/ibot loss')
    parser.add_argument('--warmup_teacher_temp', default=0.04, type=float,
                        help="""Initial value for the teacher temperature: 0.04 works well in most cases.""")
    parser.add_argument('--teacher_temp', default=0.07, type=float,
                        help="""Final value (after linear warmup)of the teacher temperature.""")
    parser.add_argument('--warmup_teacher_temp_epochs', default=-1, type=float,
                        help='Number of warmup epochs for the teacher temperature (Default: 30.0 for ImageNet).')
    parser.add_argument('--average_top_k_layers', default=1, type=int,
                        help="average over the top-k layers as targets (use -1 to indicate output of the last norm)")
    parser.add_argument('--instance_norm_target_layer', action='store_true',
                        help="apply instance normalization to target layer")
    parser.add_argument('--instance_norm_targets', action='store_true',
                        help="apply instance normalization to the final averaged targets")
    parser.add_argument('--layer_norm_targets', action='store_true',
                        help="apply layer normalization to the final averaged targets")
    parser.add_argument('--regression_loss_scale', default=None, type=float,
                        help="scale the regression loss by a factor")
    parser.add_argument('--use_smooth_l1', action='store_true',
                        help="use smooth l1 loss instead of l2 loss for regression")

    # add training parameters
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help="number of total epochs to run")
    parser.add_argument('--warmup_epochs', type=float, default=1.0, metavar='N',
                        help="epochs to warmup LR")
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help="start epoch")
    parser.add_argument('--accum_iter', default=1, type=int,
                        help="Accumulate gradient iterations (for increasing the effective batch size)")
    parser.add_argument('--moco-m', '--moco_m', default=0.994, type=float,
                        help="moco momentum of updating momentum encoder (default: 0.99)")
    parser.add_argument('--moco-m-final', '--moco_m_final', default=1.0, type=float,
                        help="moco momentum at the end of training")
    parser.add_argument('--moco-m-epochs', '--moco_m_epochs', default=0, type=int,
                        help="number of epochs for moco momentum update, 0 for disabling moco momentum update")
    parser.add_argument('--moco-m-cos', '--moco_m_cos', action='store_true',
                        help="gradually increase moco momentum to 1 with a half-cycle cosine schedule")
    parser.add_argument('--moco-m-linear', '--moco_m_linear', action='store_true',
                        help="linearly increase moco momentum")

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    parser.add_argument('--num_workers', default=12, type=int,
                        help='number of data loading workers')
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--save_freq', default=5, type=int,
                        help='the frequency to save the model')
    parser.add_argument('--n_keep', default=3, type=int,
                        help='number of checkpoints to keep')

    # optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help="weight decay (default: 0.1)")
    parser.add_argument('--weight_decay_end', type=float, default=0.1,
                        help="weight decay end (default: 0.1)")
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help="learning rate (absolute lr)")
    parser.add_argument('--beta1', type=float, default=0.9,
                        help="beta1 for AdamW optimizer")
    parser.add_argument('--beta2', type=float, default=0.95,
                        help="beta2 for AdamW optimizer")
    parser.add_argument('--blr', type=float, default=1.5e-4, metavar='LR',
                        help="base learning rate: absolute_lr = base_lr * total_batch_size / 256")
    parser.add_argument('--min_lr', type=float, default=0., metavar='LR',
                        help="lower lr bound for cyclic schedulers that hit 0")
    parser.add_argument('--clip_grad', default=None, type=float,
                        help='gradient clipping')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    return parser


def main(args):

    misc.init_distributed_mode(args)

    if misc.is_main_process():
        print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
        print("{}".format(args).replace(', ', ',\n'))

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(args.device)
    cudnn.benchmark = True

    # create dataset and loader
    train_transform = MultiCropDataAugmentation(
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        global_crops_number=args.global_crops_number,
        local_crops_number=args.local_crops_number,
        global_crops_size=args.global_crops_size,
        local_crops_size=args.local_crops_size,
    )
    if args.use_imagenet:
        train_dataset = torchvision.datasets.ImageFolder(
            root=args.imagenet_root,
            transform=train_transform,
        )
    else:
        train_dataset = SupconMultiCropDataset(
            input_filename=args.csv_path,
            root_list=args.folder_list,
            transforms=train_transform,
            num_views=args.n_img,
        )
    if misc.is_main_process():
        print(f"number of samples: {len(train_dataset)}")

    if args.distributed:
        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, num_replicas=misc.get_world_size(), rank=misc.get_rank(), shuffle=True
        )
        if misc.is_main_process():
            print("Sampler_train = %s" % str(train_sampler))
    else:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)

    n_tokens = (args.global_crops_size // args.patch_size) ** 2
    if args.mask_shape == "rand":
        mask_generator = RandomMaskingGenerator(
            input_size=args.global_crops_size // args.patch_size,
        )
    elif args.mask_shape == "block":
        if isinstance(args.mask_ratio, tuple) or isinstance(args.mask_ratio, list):
            max_mask_ratio = args.mask_ratio[1]
        else:
            max_mask_ratio = args.mask_ratio
        mask_generator = BlockMaskingGenerator(
            input_size=args.global_crops_size // args.patch_size,
            max_num_patches=int(max_mask_ratio * n_tokens),
        )
    elif args.mask_shape == "jepa_block":
        mask_generator = JepaBlockMaskingGenerator(
            input_size=args.global_crops_size // args.patch_size,
            keep_shared_min=args.mask_style == "mae",
            deterministic=False,
        )
    elif args.mask_shape == "d2v_block":
        if isinstance(args.mask_ratio, tuple) or isinstance(args.mask_ratio, list):
            mask_ratio = args.mask_ratio[0]
        else:
            mask_ratio = args.mask_ratio
        mask_generator = Data2vecBlockMaskingGenerator(
            input_size=args.global_crops_size // args.patch_size,
            mask_prob=mask_ratio,
            mask_prob_adjust=args.mask_ratio_adjust,
        )
    else:
        raise NotImplementedError

    collate_fn = partial(
        collate_data_and_cast,
        mask_ratio=args.mask_ratio,
        mask_probability=args.mask_probability,
        dtype=torch.half,   # half precision
        n_tokens=n_tokens,
        mask_first_n=args.mask_first_n,
        mask_generator=mask_generator,
        clone_batch=args.clone_batch,
        use_multi_pred=args.use_multi_pred,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=args.pin_mem, sampler=train_sampler, drop_last=True,
        collate_fn=collate_fn,
    )

    # create logger
    global_rank = misc.get_rank()
    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    model = MetaArch(args)
    model.to(device)

    # define optimizer and wrap model for distributed training
    model_without_ddp = model
    if misc.is_main_process():
        print("Model = %s" % str(model_without_ddp))

    eff_caption_batch_size = args.batch_size * args.world_size * args.accum_iter
    eff_batch_size = (eff_caption_batch_size * args.n_img * args.global_crops_number) / 2
    if args.lr is None:
        args.lr = args.blr * eff_batch_size / 256   # this is linear scaling, maybe square scaling following dinov2
    if misc.is_main_process():
        print("Effective batch size = %d, lr = %f" % (eff_batch_size, args.lr))

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))
    loss_scaler = NativeScaler()

    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model=model, data_loader=train_loader, optimizer=optimizer, device=device, epoch=epoch,
            loss_scaler=loss_scaler, log_writer=log_writer, args=args,
        )
        if args.output_dir and epoch % args.save_freq == 0:
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, n_keep=args.n_keep)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()}, 'epoch': epoch}

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # always save the latest model, for resuming training
        to_save = {
            'model': model_without_ddp.state_dict(),
            'optimizer': optimizer.state_dict(),
            'epoch': epoch,
            'scaler': loss_scaler.state_dict(),
            'args': args,
        }
        checkpoint_path = os.path.join(args.output_dir, 'epoch_last.pth')
        misc.save_on_master(to_save, checkpoint_path)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def teacher_temp_schedule(args, epoch):
    if epoch >= args.warmup_teacher_temp_epochs:
        return args.teacher_temp
    else:
        alpha = epoch / args.warmup_teacher_temp_epochs
        return args.warmup_teacher_temp * (1 - alpha) + args.teacher_temp * alpha


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int,
                    loss_scaler, log_writer=None, args=None):
    model.train()
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq
    accum_iter = args.accum_iter

    optimizer.zero_grad()   # avoid accumulating gradients

    moco_m = args.moco_m
    loader_len = len(data_loader)

    for data_iter_step, inputs_dict in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / loader_len + epoch, args)
            misc.adjust_weight_decay(optimizer, data_iter_step / loader_len + epoch, args)
        if args.moco_m_cos or args.moco_m_linear:
            # moco_m = misc.adjust_moco_momentum(data_iter_step / loader_len + epoch, args)
            moco_m = misc.adjust_moco_momentum_v2(data_iter_step / loader_len + epoch, args)

        teacher_temp = teacher_temp_schedule(args, data_iter_step / loader_len + epoch)

        for k, v in inputs_dict.items():
            if isinstance(v, torch.Tensor):
                inputs_dict[k] = v.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            loss_dict = model(inputs_dict, teacher_temp=teacher_temp)

        # multiply losses by their weights
        loss = loss_dict["loss"]

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=args.clip_grad, parameters=model.parameters(),
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        # perform teacher EMA update
        model.module.update_teacher(moco_m)     # need .module for ddp model

        for k, v in loss_dict.items():
            metric_logger.update(**{k: v.item() if isinstance(v, torch.Tensor) else v})
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes."""
            epoch_1000x = int((data_iter_step / loader_len + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
