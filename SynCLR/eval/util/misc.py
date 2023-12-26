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

import builtins
import datetime
import glob
import io
import os
import time
from collections import defaultdict, deque
from pathlib import Path
from timm.utils import get_state_dict

import torch
import torch.distributed as dist
try:
    from torch._six import inf
except:
    from torch import inf


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        log_msg = [
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    if is_main_process():
                        print(log_msg.format(
                            i, len(iterable), eta=eta_string,
                            meters=str(self),
                            time=str(iter_time), data=str(data_time),
                            memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        if is_main_process():
            print('{} Total time: {} ({:.4f} s / it)'.format(
                header, total_time_str, total_time / len(iterable)))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        force = force or (get_world_size() > 8)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def init_distributed_mode(args):
    if args.dist_on_itp:
        args.rank = int(os.environ['OMPI_COMM_WORLD_RANK'])
        args.world_size = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        args.gpu = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        args.dist_url = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(args.gpu)
        os.environ['RANK'] = str(args.rank)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # ["RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT", "LOCAL_RANK"]
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        setup_for_distributed(is_master=True)  # hack
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        args.rank, args.dist_url, args.gpu), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


class NativeScalerWithGradNormCount:
    state_dict_key = "amp_scaler"

    def __init__(self):
        self._scaler = torch.cuda.amp.GradScaler()

    def __call__(self, loss, optimizer, clip_grad=None, parameters=None, create_graph=False, update_grad=True):
        self._scaler.scale(loss).backward(create_graph=create_graph)
        if update_grad:
            if clip_grad is not None:
                assert parameters is not None
                self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                norm = torch.nn.utils.clip_grad_norm_(parameters, clip_grad)
            else:
                self._scaler.unscale_(optimizer)
                norm = get_grad_norm_(parameters)
            self._scaler.step(optimizer)
            self._scaler.update()
        else:
            norm = None
        return norm

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)


def get_grad_norm_(parameters, norm_type: float = 2.0) -> torch.Tensor:
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = [p for p in parameters if p.grad is not None]
    norm_type = float(norm_type)
    if len(parameters) == 0:
        return torch.tensor(0.)
    device = parameters[0].grad.device
    if norm_type == inf:
        total_norm = max(p.grad.detach().abs().max().to(device) for p in parameters)
    else:
        total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), norm_type).to(device) for p in parameters]), norm_type)
    return total_norm


def save_model(args, epoch, model, model_without_ddp, optimizer, loss_scaler, model_ema=None, n_keep=10):
    output_dir = Path(args.output_dir)
    epoch_name = str(epoch)

    if loss_scaler is not None:
        checkpoint_paths = [output_dir / ('checkpoint-%s.pth' % epoch_name)]
        for checkpoint_path in checkpoint_paths:
            to_save = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }

            if model_ema is not None:
                to_save['model_ema'] = get_state_dict(model_ema)

            save_on_master(to_save, checkpoint_path)
            del to_save
    else:
        client_state = {'epoch': epoch}
        model.save_checkpoint(save_dir=args.output_dir, tag="checkpoint-%s" % epoch_name, client_state=client_state)

    # keep n_keep checkpoints
    if is_main_process():
        model_list = glob.glob(os.path.join(args.output_dir, 'checkpoint-*.pth'))
        if len(model_list) > n_keep:
            epochs = [os.path.basename(m).split('-')[-1].split('.')[0] for m in model_list]
            epochs = [int(e) for e in epochs]
            epochs.sort(reverse=True)
            for e in range(n_keep, len(epochs)):
                name = os.path.join(args.output_dir, 'checkpoint-%s.pth' % str(epochs[e]))
                cmd = 'rm %s' % name
                os.system(cmd)


def _load_checkpoint_for_ema(model_ema, checkpoint):
    """
    Workaround for ModelEma._load_checkpoint to accept an already-loaded object
    """
    mem_file = io.BytesIO()
    temp = {}
    temp['state_dict_ema'] = checkpoint
    torch.save(temp, mem_file)
    mem_file.seek(0)
    model_ema._load_checkpoint(mem_file)
    # mem_file = io.BytesIO()
    # torch.save(checkpoint, mem_file)
    # mem_file.seek(0)
    # model_ema._load_checkpoint(mem_file)


def load_model(args, model_without_ddp, optimizer, loss_scaler, model_ema=None):
    if args.resume:
        if args.resume.startswith('https'):
            checkpoint = torch.hub.load_state_dict_from_url(
                args.resume, map_location='cpu', check_hash=True)
        else:
            checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])

        try:
            if args.model_ema:
                _load_checkpoint_for_ema(model_ema, checkpoint['model_ema'])
                print("With EMA resumed!")
        except:
            print('No ema in namespace')

        print("Resume checkpoint %s" % args.resume)
        if 'optimizer' in checkpoint and 'epoch' in checkpoint and not (hasattr(args, 'eval') and args.eval):
            optimizer.load_state_dict(checkpoint['optimizer'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
            print("With optim & sched!")


def all_reduce_mean(x):
    world_size = get_world_size()
    if world_size > 1:
        x_reduce = torch.tensor(x).cuda()
        dist.all_reduce(x_reduce)
        x_reduce /= world_size
        return x_reduce.item()
    else:
        return x


def filter_model_dict(ckpt, use_ema=False):

    print('use ema:', use_ema)

    keyword = 'target_encoder.' if use_ema else 'online_encoder.'

    is_mae = True
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith(keyword):
            is_mae = False
            new_k = k.replace(keyword, '')
            new_ckpt[new_k] = v

    return ckpt if is_mae else new_ckpt


@torch.no_grad()
def normalize_kappa(kappa):
    n_dims = len(kappa.shape)
    assert n_dims <= 2, f'kappa dimension: {n_dims}'
    total = torch.numel(kappa) * get_world_size()

    kappa_sum = kappa.sum().clone()
    dist.all_reduce(kappa_sum, op=dist.ReduceOp.SUM)
    kappa_ = kappa * total / kappa_sum
    return kappa_


@torch.no_grad()
def get_kappa_stats(kappa):
    k_min, k_max, k_mean = kappa.min(), kappa.max(), kappa.mean()

    all_k_min = concat_all_gather(k_min)
    k_min = all_k_min.min()
    all_k_max = concat_all_gather(k_max)
    k_max = all_k_max.max()
    dist.all_reduce(k_mean, op=dist.ReduceOp.SUM)
    k_mean /= get_world_size()
    return k_min, k_max, k_mean


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    # robust to a single value gathering
    if len(tensors_gather[0].shape) == 0:
        tensors_gather = [x.view(1) for x in tensors_gather]

    output = torch.cat(tensors_gather, dim=0)
    return output


class OnlineKnn(torch.nn.Module):

    def __init__(self, size=1024 * 30, dim=256):
        super().__init__()

        self.size = size
        self.dim = dim
        self.register_buffer("queue", torch.randn(dim, size))
        self.register_buffer("label", torch.zeros(size, dtype=torch.long))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def dequeue_and_enqueue(self, feats, labels):

        feats = concat_all_gather(feats)
        labels = concat_all_gather(labels)
        batch_size = feats.shape[0]

        ptr = int(self.queue_ptr)

        copy_ids = torch.arange(batch_size).cuda()
        copy_ids += ptr
        copy_ids = torch.fmod(copy_ids, self.size).long()
        feats = feats.float()
        self.queue.index_copy_(1, copy_ids, feats.T)
        self.label.index_copy_(0, copy_ids, labels)

        # assert self.size % batch_size == 0
        # self.queue[:, ptr:ptr + batch_size] = feats.T
        # self.label[ptr:ptr + batch_size] = labels

        ptr = (ptr + batch_size) % self.size
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def knn_classify(self, test_feats, test_labels, k, t, num_classes):
        top1, top5, total = 0.0, 0.0, 0
        num_test_images, num_chunks = test_labels.shape[0], 10
        imgs_per_chunk = num_test_images // num_chunks
        retrieval_one_hot = torch.zeros(k, num_classes).to(test_feats.device)

        for idx in range(0, num_test_images, imgs_per_chunk):
            features = test_feats[idx: min((idx + imgs_per_chunk), num_test_images), :]
            targets = test_labels[idx: min((idx + imgs_per_chunk), num_test_images)]
            batch_size = targets.shape[0]

            similarity = torch.mm(features, self.queue)
            distances, indices = similarity.topk(k, largest=True, sorted=True)
            candidates = self.label.view(1, -1).expand(batch_size, -1)
            retrieved_neighbors = torch.gather(candidates, 1, indices)

            retrieval_one_hot.resize_(batch_size * k, num_classes).zero_()
            retrieval_one_hot.scatter_(1, retrieved_neighbors.view(-1, 1), 1)

            distances_transform = distances.clone().div_(t).exp_()

            probs = torch.sum(
                torch.mul(
                    retrieval_one_hot.view(batch_size, -1, num_classes),
                    distances_transform.view(batch_size, -1, 1),
                ),
                1,
            )
            _, predictions = probs.sort(1, True)

            correct = predictions.eq(targets.data.view(-1, 1))
            top1 = top1 + correct.narrow(1, 0, 1).sum().item()
            top5 = top5 + correct.narrow(1, 0, min(5, k)).sum().item()
            total += targets.size(0)

        top1 = top1 * 100.0 / total
        top5 = top5 * 100.0 / total
        return top1, top5
