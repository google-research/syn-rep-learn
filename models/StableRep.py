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
import timm

from collections import OrderedDict


class VisionEncoder(nn.Module):
    def __init__(self,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # ssl
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 **kwargs,
                 ):
        super().__init__()

        self.vision_width = vision_width
        self.visual = vision_model

        self.image_mlp = nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(vision_width, ssl_mlp_dim)),
            ("bn1", nn.SyncBatchNorm(ssl_mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(ssl_mlp_dim, ssl_mlp_dim)),
            ("bn2", nn.SyncBatchNorm(ssl_mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(ssl_mlp_dim, ssl_emb_dim)),
        ]))

    def forward(self, img_cat, label, *args, **kwargs):
        n_caption = img_cat.size(0)
        augs = torch.split(img_cat, 3, dim=1)
        n_views = len(augs)

        # do separate BN
        images = torch.cat(augs, dim=0)
        feats = self.visual(images)
        feats = torch.split(feats, n_caption, dim=0)
        res = [self.image_mlp(feat) for feat in feats]
        h = torch.cat(res, dim=0)

        # # can do a global BN instead of separate BN
        # images = torch.cat(augs, dim=0)
        # h = self.visual(images)
        # h = self.image_mlp(h)

        label = label.view(-1, 1)
        label_expand = label.repeat(n_views, 1).squeeze()

        return {'feats': h,
                'labels': label_expand}


def stablerep_vit_small_patch16(ssl_mlp_dim, ssl_emb_dim, **kwargs):
    vision_model = timm.create_model('vit_small_patch16_224', num_classes=0)
    model = VisionEncoder(vision_width=384, vision_model=vision_model,
                          ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)

    return model


def stablerep_vit_base_patch16(ssl_mlp_dim, ssl_emb_dim, **kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = VisionEncoder(vision_width=768, vision_model=vision_model,
                          ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)

    return model


def stablerep_vit_large_patch16(ssl_mlp_dim, ssl_emb_dim, **kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = VisionEncoder(vision_width=1024, vision_model=vision_model,
                          ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim, **kwargs)

    return model


model_dict = {
    'small': stablerep_vit_small_patch16,
    'base': stablerep_vit_base_patch16,
    'large': stablerep_vit_large_patch16,
}
