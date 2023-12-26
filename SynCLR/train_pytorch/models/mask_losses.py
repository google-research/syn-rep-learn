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

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed.nn
import torch.distributed as dist

from util import misc


try:
    from xformers.ops import cross_entropy

    def lossfunc(t, s, temp):
        s = s.float()
        t = t.float()
        if s.ndim == 2:
            return -cross_entropy(s.unsqueeze(0), t.unsqueeze(0), temp, bw_inplace=True).squeeze(0)
        elif s.ndim == 3:
            return -cross_entropy(s, t, temp, bw_inplace=True)

except ImportError:

    def lossfunc(t, s, temp):
        return torch.sum(t * F.log_softmax(s / temp, dim=-1), dim=-1)


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output


class SupConMomentumLoss(nn.Module):
    """
    SupCon loss: https://arxiv.org/abs/2004.11362, with momentum encoder
    """
    def __init__(
        self,
        temperature=0.1,
        avoid_local_global_same=True,
    ):
        super(SupConMomentumLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

        # for each local crop, whether we exclude the global crop from the same image
        self.avoid_local_global_same = avoid_local_global_same

    def forward(self, outputs):
        q = outputs['student_contrast_cls']
        k = outputs['teacher_contrast_cls']
        s_labels = outputs['student_labels']
        t_labels = outputs['teacher_labels']

        device = (torch.device('cuda')
                  if q.is_cuda
                  else torch.device('cpu'))

        q = nn.functional.normalize(q, dim=1)
        k = nn.functional.normalize(k, dim=1)

        local_batch_size = n_img = k.size(0)    # assume one teacher crop per image

        # all_k = k
        # all_labels = t_labels
        all_k = concat_all_gather(k)
        all_labels = concat_all_gather(t_labels)

        # compute the mask based on labels
        if local_batch_size != self.last_local_batch_size:
            mask = torch.eq(s_labels.view(-1, 1), all_labels.contiguous().view(1, -1)).float().to(device)
            exclude_index = torch.arange(n_img).view(-1, 1).to(device)
            if self.avoid_local_global_same:
                exclude_index = exclude_index.repeat(q.shape[0] // n_img, 1)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                exclude_index + local_batch_size * misc.get_rank(),
                0
            )
            self.last_local_batch_size = local_batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask
        logits = torch.matmul(q, all_k.T) / self.temperature

        # for numerical stability
        logits_max, _ = torch.max(logits, dim=1, keepdim=True)
        logits = logits - logits_max.detach()

        # compute log_prob
        exp_logits = torch.exp(logits) * self.logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        global_crop_contrast_loss = - mean_log_prob_pos[:n_img].mean()
        if q.shape[0] > local_batch_size:
            local_crop_contrast_loss = - mean_log_prob_pos[n_img:].mean()
        else:
            local_crop_contrast_loss = 0    # if no local patches, set loss to zero
        total_contrast_loss = global_crop_contrast_loss + local_crop_contrast_loss

        return {
            'global_crop_contrast_loss': global_crop_contrast_loss,
            'local_crop_contrast_loss': local_crop_contrast_loss,
            'total_contrast_loss': total_contrast_loss,
        }


class DINOLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        student_temp=0.1,
        center_momentum=0.9,
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_output = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_output, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        return F.softmax((teacher_output - self.center) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_iterations=3):
        teacher_output = teacher_output.float()
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        B = Q.shape[1] * world_size  # number of samples to assign
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_output_list, teacher_out_softmaxed_centered_list):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        # TODO: Use cross_entropy_distribution here
        total_loss = 0
        for s in student_output_list:
            lsm = F.log_softmax(s / self.student_temp, dim=-1)
            for t in teacher_out_softmaxed_centered_list:
                loss = torch.sum(t * lsm, dim=-1)
                total_loss -= loss.mean()
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        self.reduce_center_update(teacher_output)

    @torch.no_grad()
    def reduce_center_update(self, teacher_output):
        self.updated = False
        self.len_teacher_output = len(teacher_output)
        self.async_batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_output * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


class iBOTPatchLoss(nn.Module):
    def __init__(self, patch_out_dim, student_temp=0.1, center_momentum=0.9):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, 1, patch_out_dim))
        self.updated = True
        self.reduce_handle = None
        self.len_teacher_patch_tokens = None
        self.async_batch_center = None

    @torch.no_grad()
    def softmax_center_teacher(self, teacher_patch_tokens, teacher_temp):
        self.apply_center_update()
        # teacher centering and sharpening
        #
        # WARNING:
        #   as self.center is a float32, everything gets casted to float32 afterwards
        #
        # teacher_patch_tokens = teacher_patch_tokens.float()
        # return F.softmax((teacher_patch_tokens.sub_(self.center.to(teacher_patch_tokens.dtype))).mul_(1 / teacher_temp), dim=-1)

        return F.softmax((teacher_patch_tokens - self.center) / teacher_temp, dim=-1)

        # this is experimental, keep everything in float16 and let's see what happens:
        # return F.softmax((teacher_patch_tokens.sub_(self.center)) / teacher_temp, dim=-1)

    @torch.no_grad()
    def sinkhorn_knopp_teacher(self, teacher_output, teacher_temp, n_masked_patches_tensor, n_iterations=3):
        teacher_output = teacher_output.float()
        # world_size = dist.get_world_size() if dist.is_initialized() else 1
        Q = torch.exp(teacher_output / teacher_temp).t()  # Q is K-by-B for consistency with notations from our paper
        # B = Q.shape[1] * world_size # number of samples to assign
        B = n_masked_patches_tensor
        dist.all_reduce(B)
        K = Q.shape[0]  # how many prototypes

        # make the matrix sums to 1
        sum_Q = torch.sum(Q)
        if dist.is_initialized():
            dist.all_reduce(sum_Q)
        Q /= sum_Q

        for it in range(n_iterations):
            # normalize each row: total weight per prototype must be 1/K
            sum_of_rows = torch.sum(Q, dim=1, keepdim=True)
            if dist.is_initialized():
                dist.all_reduce(sum_of_rows)
            Q /= sum_of_rows
            Q /= K

            # normalize each column: total weight per sample must be 1/B
            Q /= torch.sum(Q, dim=0, keepdim=True)
            Q /= B

        Q *= B  # the columns must sum to 1 so that Q is an assignment
        return Q.t()

    def forward(self, student_patch_tokens, teacher_patch_tokens, student_masks_flat):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        student_patch_tokens: (B, N, D) tensor
        teacher_patch_tokens: (B, N, D) tensor
        student_masks_flat: (B, N) tensor
        """
        t = teacher_patch_tokens
        s = student_patch_tokens
        loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = torch.sum(loss * student_masks_flat.float(), dim=-1) / student_masks_flat.sum(dim=-1).clamp(min=1.0)
        return -loss.mean()

    def forward_masked(
        self,
        student_patch_tokens_masked,
        teacher_patch_tokens_masked,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
        n_masked_samples=None,
    ):
        t = teacher_patch_tokens_masked
        s = student_patch_tokens_masked
        # loss = torch.sum(t * F.log_softmax(s / self.student_temp, dim=-1), dim=-1)
        loss = lossfunc(t, s, self.student_temp)
        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight
        normalizer = n_masked_samples if n_masked_samples is not None else student_masks_flat.shape[0]
        return -loss.sum() / normalizer

    @torch.no_grad()
    def update_center(self, teacher_patch_tokens):
        self.reduce_center_update(teacher_patch_tokens)

    @torch.no_grad()
    def reduce_center_update(self, teacher_patch_tokens):
        self.updated = False
        self.len_teacher_patch_tokens = len(teacher_patch_tokens)
        self.async_batch_center = torch.sum(teacher_patch_tokens.mean(1), dim=0, keepdim=True)
        if dist.is_initialized():
            self.reduce_handle = dist.all_reduce(self.async_batch_center, async_op=True)

    @torch.no_grad()
    def apply_center_update(self):
        if self.updated is False:
            world_size = dist.get_world_size() if dist.is_initialized() else 1

            if self.reduce_handle is not None:
                self.reduce_handle.wait()
            _t = self.async_batch_center / (self.len_teacher_patch_tokens * world_size)

            self.center = self.center * self.center_momentum + _t * (1 - self.center_momentum)

            self.updated = True


class MAEFeatLoss(nn.Module):
    """feature predictor loss for MAE style encoding"""
    def __init__(
        self,
        loss_scale=None,
        use_smooth_l1=False,
    ):
        super().__init__()
        self.loss_scale = loss_scale
        self.use_smooth_l1 = use_smooth_l1

    def forward(
        self,
        x,
        y,
        student_masks_flat,
        n_masked_patches=None,
        masks_weight=None,
        n_masked_samples=None,
    ):
        x = x.view(-1, x.size(-1)).float()
        y = y.view(-1, x.size(-1)).float()

        if self.use_smooth_l1:
            loss = F.smooth_l1_loss(x, y, reduction='none')
        else:
            loss = F.mse_loss(x, y, reduction='none')
        loss = loss.mean(-1)
        scale = self.loss_scale if self.loss_scale is not None else 1.0

        if masks_weight is None:
            masks_weight = (
                (1 / student_masks_flat.sum(-1).clamp(min=1.0))
                .unsqueeze(-1)
                .expand_as(student_masks_flat)[student_masks_flat]
            )
        if n_masked_patches is not None:
            loss = loss[:n_masked_patches]
        loss = loss * masks_weight * scale
        normalizer = n_masked_samples if n_masked_samples is not None else student_masks_flat.shape[0]
        return loss.sum() / normalizer


if __name__ == "__main__":
    loss_fn = SupConMomentumLoss(temperature=0.1, avoid_local_global_same=False)
    outputs = dict()
    outputs['student_contrast_cls'] = torch.rand(12, 64)
    outputs['teacher_contrast_cls'] = torch.rand(4, 64)
    outputs['student_labels'] = torch.Tensor([1, 1, 2, 2]).repeat(3)
    outputs['teacher_labels'] = torch.Tensor([1, 1, 2, 2])

    loss_dict = loss_fn(outputs)
    print(loss_dict['global_crop_contrast_loss'])
    print(loss_dict['local_crop_contrast_loss'])
    print(loss_dict['total_contrast_loss'])
