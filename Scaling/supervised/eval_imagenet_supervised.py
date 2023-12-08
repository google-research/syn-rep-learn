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
import os
import time

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms

import models_vit

from torch import nn
from torchvision import datasets


def get_args_parser():
    parser = argparse.ArgumentParser(description='Supervised Evaluation', add_help=False)
    parser.add_argument('--data-path', default='dataset/imagenet', type=str,
                        help='dataset path')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='vit_base_patch16_224',
                        help='model architecture')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 8)')
    parser.add_argument('-b', '--batch-size', default=128, type=int,
                        metavar='N')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='path to checkpoint to eval (default: none)')
    parser.add_argument('--gpu', default=0, type=int,
                        help='GPU id to use.')
    return parser


def main(args):
    # create model
    print("=> creating model '{}'".format(args.arch))
    model = models_vit.create_model(args.arch, num_classes=1000)
    for name, param in model.named_parameters():
        param.requires_grad = False
    loc = 'cuda:{}'.format(args.gpu)
    state_dict = torch.load(args.ckpt, map_location=loc)
    msg = model.load_state_dict(state_dict, strict=True)
    print(msg)
    cudnn.benchmark = True

    normalize = transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                     std=[0.5, 0.5, 0.5])
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        lambda x: x.convert('RGB'),
        transforms.ToTensor(),
        normalize,
    ])

    data_path_val = os.path.join(args.data_path, 'val')
    val_dataset = datasets.ImageFolder(data_path_val, val_transform)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    val_stats = validate(val_loader, model, args)
    print('checkpoint: {}'.format(args.ckpt))
    print('Top-1 Accuracy: {}'.format(val_stats['acc1']))
    print('Loss: {}'.format(val_stats['loss']))


def validate(val_loader, model, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(val_loader),
        [batch_time, losses, top1, top5],
        prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(val_loader):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)
            if torch.cuda.is_available():
                target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            outputs = model(images)
            outputs = outputs.detach()
            loss_batch = nn.CrossEntropyLoss()(outputs, target)
            acc1_batch, acc5_batch = accuracy(outputs, target, topk=(1, 5))

            # measure accuracy and record loss
            losses.update(loss_batch.item(), images.size(0))
            top1.update(acc1_batch.item(), images.size(0))
            top5.update(acc5_batch.item(), images.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)
    return {'acc1': top1.avg, 'acc5': top5.avg, 'loss': losses.avg}


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Supervised evaluation', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
