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
import math
import numpy as np
import os
import sys
import time
from pathlib import Path
from typing import Iterable

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms

from dataset.util import GaussianBlur, DownSampleAndUpsample
from dataset.data import SupconDataset
from models.losses import MultiPosConLoss, MultiPosConLossMM
from models.StableRep import model_dict as v_model_dict
from models.StableRepMM import model_dict as vt_model_dict

import util.misc as misc
from util.misc import NativeScalerWithGradNormCount as NativeScaler


def get_args_parser():
    parser = argparse.ArgumentParser('StableRep pre-training', add_help=False)
    parser.add_argument('--epochs', default=15, type=int)

    # Model parameters
    parser.add_argument('--model', default='base', type=str,
                        help='Name of model to train')
    parser.add_argument('--add_language', action='store_true',
                        help='adding language to the model')
    parser.add_argument('--tokenizer', default='CLIP', type=str, choices=['CLIP'],
                        help='tokenization choice (only CLIP here)')

    # add self-supervised learning parameters
    parser.add_argument('--ssl-mlp-dim', default=4096, type=int,
                        help='hidden dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-emb-dim', default=256, type=int,
                        help='output embed dim of SimCLR mlp projection head')
    parser.add_argument('--ssl-scale', default=1.0, type=float,
                        help='loss scale for SimCLR objective')
    parser.add_argument('--ssl-temp', default=0.1, type=float,
                        help='softmax temperature for SimCLR objective')
    parser.add_argument('--ssl-temp-cos', action='store_true',
                        help='gradually increase the ssl temperature')
    parser.add_argument('--ssl-temp-min', default=0.05, type=float,
                        help='minimum temperature of the cosine cycle')
    parser.add_argument('--ssl-temp-max', default=0.1, type=float,
                        help='maximum temperature of the cosine cycle')
    parser.add_argument('--ssl-w1', default=1.0, type=float,
                        help='weight for image multi-positive loss')
    parser.add_argument('--ssl-w2', default=1.0, type=float,
                        help='weight for image text multi-positive loss')
    parser.add_argument('--vl-projection', default='linear', type=str,
                        choices=['linear', 'mlp'], help='projection head type')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.1,
                        help='weight decay (default: 0.05)')
    parser.add_argument('--lr', type=float, default=None,
                        help='learning rate (absolute lr)')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='beta1 for AdamW optimizer')
    parser.add_argument('--beta2', type=float, default=0.95,
                        help='beta2 for AdamW optimizer')
    parser.add_argument('--blr', type=float, default=1e-3,
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--min_lr', type=float, default=0.,
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=float, default=1.0,
                        help='epochs to warmup LR')

    parser.add_argument('--output_dir', default='./output_dir',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='./output_dir',
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')

    parser.add_argument('--start_epoch', default=0, type=int,
                        help='start epoch')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient transfer to GPU.')
    parser.set_defaults(pin_mem=True)

    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--save_freq', default=5, type=int,
                        help='the frequency to save the model')
    parser.add_argument('--n_keep', default=3, type=int,
                        help='number of checkpoints to keep')

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=-1, type=int)
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')

    # Dataset parameters
    parser.add_argument('--csv_path', default='./data/MAE/MAE_train.csv',
                        help='csv file path')
    parser.add_argument('--folder_list', nargs='+',
                        help='A list of items')
    parser.add_argument('--n_img', type=int, default=1,
                        help='number of images per caption sample, default: 1')
    parser.add_argument('--num_crop', type=int, default=1,
                        help='number of crops per images')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per GPU.')
    parser.add_argument('--weak_aug', action='store_true',
                        help='use weak augmentation for each image')

    # downsample aug parameter
    parser.add_argument('--downsample', action='store_true',
                        help='randomly downsample images')
    parser.add_argument('--downsample_prob', default=0.05, type=float,
                        help='prob for applying this augmentation')
    parser.add_argument('--down_res', default=None, nargs='+',
                        help='A list of downsample resolutions')
    parser.add_argument('--down_prob', default=None, nargs='+',
                        help='A list of downsample probabilities (corresponds to each resolution),'
                             'can be un-normalized probabilities')
    return parser


def main_print(obj):
    if misc.is_main_process():
        print(obj)


def main(args):
    misc.init_distributed_mode(args)

    # ======= adapt args =======
    if args.down_res:
        args.down_res = [int(x) for x in args.down_res]
    if args.down_prob:
        args.down_prob = [float(x) for x in args.down_prob]
    # ======= adapt args =======

    main_print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    main_print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    cudnn.benchmark = True

    # specify data loading
    if args.weak_aug:
        main_print('using weak augmentation')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])
    else:
        main_print('using strong augmentation')
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])])

    if args.downsample:
        main_print('add downsample augmentation')
        train_transform = transforms.Compose([
            transforms.RandomApply([DownSampleAndUpsample(down_res=args.down_res, p=args.down_prob)],
                                   p=args.downsample_prob),
            train_transform])

    main_print(('csv path:', args.csv_path))
    main_print(('data folder list:', args.folder_list))

    train_dataset = SupconDataset(
        input_filename=args.csv_path,
        root_list=args.folder_list,
        transforms=train_transform,
        num_views=args.n_img,
        num_crop=args.num_crop,
        tokenizer=args.tokenizer if args.add_language else None,
    )
    main_print(len(train_dataset))

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=args.pin_mem, sampler=train_sampler, drop_last=True)

    global_rank = misc.get_rank()

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    if not args.add_language:
        # StableRep w/o language
        model = v_model_dict[args.model](ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim)
        criterion = MultiPosConLoss(temperature=args.ssl_temp)
    else:
        # StableRep w/ language
        model = vt_model_dict[args.model](ssl_mlp_dim=args.ssl_mlp_dim, ssl_emb_dim=args.ssl_emb_dim,
                                          vl_projection=args.vl_projection)
        criterion = MultiPosConLossMM(temperature=args.ssl_temp, w1=args.ssl_w1, w2=args.ssl_w2)

    model = model.to(device)
    criterion = criterion.to(device)
    model_without_ddp = model

    if args.lr is None:  # only base_lr is specified
        eff_batch_size = args.batch_size * misc.get_world_size()
        args.lr = args.blr * eff_batch_size * args.n_img / 256
        args.lr = args.lr * args.num_crop / 2  # previous line assumes num_crop=2

    main_print("lr: %.3e" % args.lr)

    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],
                                                          find_unused_parameters=False)
        model_without_ddp = model.module

    param_groups = misc.add_weight_decay(model, args.weight_decay)
    optimizer = torch.optim.AdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2))
    loss_scaler = NativeScaler()

    # resume model if needed
    misc.load_model(args=args, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler)

    main_print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        if args.distributed:
            train_sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, train_loader,
            optimizer, device, epoch, loss_scaler, criterion,
            log_writer=log_writer,
            args=args
        )

        if args.output_dir and (epoch % args.save_freq == 0 or epoch + 1 == args.epochs):
            misc.save_model(
                args=args, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer,
                loss_scaler=loss_scaler, epoch=epoch, n_keep=args.n_keep)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     'epoch': epoch, }

        if args.output_dir and misc.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")

        # always save the last model
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
    main_print('Training time {}'.format(total_time_str))


def train_one_epoch(model: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, loss_fn=None,
                    log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = args.print_freq

    optimizer.zero_grad()

    # this is a pseudo label to index samples for loss function
    label_input = torch.arange(args.batch_size).to(device) + \
        args.batch_size * misc.get_rank()
    text_input = None

    for data_iter_step, data in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        loader_len = len(data_loader)
        misc.adjust_learning_rate(optimizer, data_iter_step / loader_len + epoch, args)

        img_input = data[0].to(device, non_blocking=True)
        if args.add_language:
            text_input = data[1].to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            outputs = model(img_input, label_input, text_input)
        loss_dict = loss_fn(outputs)
        loss = loss_dict['loss']

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss_scaler(loss, optimizer, parameters=model.parameters())
        optimizer.zero_grad()

        # clamp logit scale for image-text contrast
        if args.add_language:
            misc.get_model(model).logit_scale.data.clamp_(0, 4.6052)
            logit_scale = misc.get_model(model).logit_scale.exp().item()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)

        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(lr=lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / loader_len + epoch) * 1000)
            log_writer.add_scalar('train_loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', lr, epoch_1000x)
            if args.add_language:
                log_writer.add_scalar('logit', logit_scale, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    main_print(("Averaged stats:", metric_logger))
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)
