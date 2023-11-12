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

from util import misc


def compute_cross_entropy(p, q):
    q = F.log_softmax(q, dim=-1)
    loss = torch.sum(p * q, dim=-1)
    return - loss.mean()


def stablize_logits(logits):
    logits_max, _ = torch.max(logits, dim=-1, keepdim=True)
    logits = logits - logits_max.detach()
    return logits


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


class MultiPosConLoss(nn.Module):
    """
    Multi-Positive Contrastive Loss: https://arxiv.org/pdf/2306.00984.pdf
    """

    def __init__(self, temperature=0.1):
        super(MultiPosConLoss, self).__init__()
        self.temperature = temperature
        self.logits_mask = None
        self.mask = None
        self.last_local_batch_size = None

    def set_temperature(self, temp=0.1):
        self.temperature = temp

    def forward(self, outputs):
        feats = outputs['feats']    # feats shape: [B, D]
        labels = outputs['labels']    # labels shape: [B]

        device = (torch.device('cuda')
                  if feats.is_cuda
                  else torch.device('cpu'))

        feats = F.normalize(feats, dim=-1, p=2)
        local_batch_size = feats.size(0)

        all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)
        all_labels = concat_all_gather(labels)  # no gradient gather

        # compute the mask based on labels
        if local_batch_size != self.last_local_batch_size:
            mask = torch.eq(labels.view(-1, 1),
                            all_labels.contiguous().view(1, -1)).float().to(device)
            self.logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(mask.shape[0]).view(-1, 1).to(device) +
                local_batch_size * misc.get_rank(),
                0
            )

            self.last_local_batch_size = local_batch_size
            self.mask = mask * self.logits_mask

        mask = self.mask

        # compute logits
        logits = torch.matmul(feats, all_feats.T) / self.temperature
        logits = logits - (1 - self.logits_mask) * 1e9

        # optional: minus the largest logit to stablize logits
        logits = stablize_logits(logits)

        # compute ground-truth distribution
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        loss = compute_cross_entropy(p, logits)

        return {'loss': loss, 'image_loss': loss}


class MultiPosConLossMM(nn.Module):
    """Multi-positive contrastive loss, when multiple images corresponds to the same texts"""
    def __init__(self, temperature=0.1, w1=1.0, w2=1.0):
        """
        Args:
            temperature: temperature for contrastive loss
            w1: weight for the image contrastive part
            w2: weight for the image-text part
        """
        super(MultiPosConLossMM, self).__init__()
        self.temperature = temperature
        self.w1 = w1
        self.w2 = w2
        self.last_local_batch_size = None
        self.v_label_matrix = None
        self.t_label_matrix = None
        self.mask = None
        self.logits_mask = None

    def forward(self, outputs):
        v_feats = outputs['image_emb']
        t_feats = outputs['text_emb']
        v_labels = outputs['image_labels']
        t_labels = outputs['text_labels']
        logit_scale = outputs['logit_scale']
        device = (torch.device('cuda')
                  if v_feats.is_cuda
                  else torch.device('cpu'))

        v_feats = F.normalize(v_feats, dim=-1, p=2)
        t_feats = F.normalize(t_feats, dim=-1, p=2)

        v_local_batch_size = v_feats.size(0)
        t_local_batch_size = t_feats.size(0)

        all_v_feats = torch.cat(torch.distributed.nn.all_gather(v_feats), dim=0)
        all_t_feats = torch.cat(torch.distributed.nn.all_gather(t_feats), dim=0)

        # compute the logits for image-text contrasting
        logits_v = logit_scale * torch.matmul(v_feats, all_t_feats.T)
        logits_t = logit_scale * torch.matmul(t_feats, all_v_feats.T)

        # compute the logits for image-only contrasting
        feats = outputs['image_feats']
        feats = F.normalize(feats, dim=-1, p=2)
        all_feats = torch.cat(torch.distributed.nn.all_gather(feats), dim=0)
        logits = torch.matmul(feats, all_feats.T) / self.temperature

        # Create label matrix, since in our specific case the
        # label matrix in side each batch is the same, so
        # we can just create it once and reuse it. For other
        # cases, user need to compute it for each batch
        if v_local_batch_size != self.last_local_batch_size:
            all_v_labels = concat_all_gather(v_labels)
            all_t_labels = concat_all_gather(t_labels)
            all_v_labels = all_v_labels.contiguous().view(1, -1)
            all_t_labels = all_t_labels.contiguous().view(1, -1)

            # mask matrix for image-text contrastive loss
            self.v_label_matrix = torch.eq(v_labels.view(-1, 1),
                                           all_t_labels).float().to(device)
            self.t_label_matrix = torch.eq(t_labels.view(-1, 1),
                                           all_v_labels).float().to(device)

            # mask matrix for image supervised contrastive loss
            self.mask = torch.eq(v_labels.view(-1, 1), all_v_labels).float().to(device)
            self.logits_mask = torch.scatter(
                torch.ones_like(self.mask),
                1,
                torch.arange(self.mask.shape[0]).view(-1, 1).to(device) +
                v_local_batch_size * misc.get_rank(),
                0
            )
            self.mask = self.mask * self.logits_mask

            self.last_local_batch_size = v_local_batch_size

        # image only loss
        mask = self.mask
        p = mask / mask.sum(1, keepdim=True).clamp(min=1.0)
        logits = logits - (1 - self.logits_mask) * 1e9
        logits = stablize_logits(logits)
        img_loss = compute_cross_entropy(p, logits)

        # image text loss
        v_mask = self.v_label_matrix
        p_v = v_mask / v_mask.sum(1, keepdim=True).clamp(min=1.0)
        t_mask = self.t_label_matrix
        p_t = t_mask / t_mask.sum(1, keepdim=True).clamp(min=1.0)
        img_txt_loss = (compute_cross_entropy(p_v, logits_v) + compute_cross_entropy(p_t, logits_t)) / 2

        # total loss
        loss = self.w1 * img_loss + self.w2 * img_txt_loss

        return {'loss': loss,
                'image_loss': img_loss,
                'img_txt_loss': img_txt_loss}
