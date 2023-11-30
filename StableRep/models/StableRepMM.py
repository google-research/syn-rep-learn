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
import numpy as np

from collections import OrderedDict


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        ret = super().forward(x.type(torch.float32))
        return ret.type(orig_type)


class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) \
            if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class Transformer(nn.Module):
    def __init__(self, width: int, layers: int, heads: int,
                 attn_mask: torch.Tensor = None):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.Sequential(*[ResidualAttentionBlock(width, heads, attn_mask)
                                         for _ in range(layers)])

    def forward(self, x: torch.Tensor):
        return self.resblocks(x)


class VisionTextEncoder(nn.Module):
    def __init__(self,
                 # vision
                 vision_width: int,
                 vision_model: nn.Module,
                 # ssl
                 ssl_mlp_dim: int,
                 ssl_emb_dim: int,
                 # text
                 context_length: int,
                 vocab_size: int,
                 transformer_width: int,
                 transformer_heads: int,
                 transformer_layers: int,
                 vl_projection: str,  # 'linear' or 'mlp'
                 embed_dim: int,
                 **kwargs,
                 ):
        super().__init__()

        self.context_length = context_length

        # image
        self.vision_width = vision_width
        self.visual = vision_model
        self.image_mlp = self._build_mlp(in_dim=vision_width,
                                         mlp_dim=ssl_mlp_dim,
                                         out_dim=ssl_emb_dim)

        # text
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=self.build_attention_mask(),
        )

        self.vocab_size = vocab_size
        self.token_embedding = nn.Embedding(vocab_size, transformer_width)
        self.positional_embedding = nn.Parameter(
            torch.empty(self.context_length, transformer_width))
        self.ln_final = LayerNorm(transformer_width)

        # image-text projection
        self.vl_projection = vl_projection
        if self.vl_projection == 'linear':
            self.image_projection = nn.Parameter(
                torch.empty(vision_width, embed_dim))
            self.text_projection = nn.Parameter(
                torch.empty(transformer_width, embed_dim))
        elif self.vl_projection == 'mlp':
            self.image_projection = self._build_mlp(in_dim=vision_width,
                                                    mlp_dim=ssl_mlp_dim,
                                                    out_dim=ssl_emb_dim)
            self.text_projection = self._build_mlp(in_dim=transformer_width,
                                                   mlp_dim=ssl_mlp_dim,
                                                   out_dim=ssl_emb_dim)
        else:
            raise ValueError(f'Invalid vl_projection: {self.vl_projection}')
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.initialize_parameters()

    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)

        proj_std = ((self.transformer.width ** -0.5) *
                    ((2 * self.transformer.layers) ** -0.5))
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        if self.vl_projection == 'linear':
            nn.init.normal_(self.image_projection, std=self.vision_width ** -0.5)
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the
        # vision tokens pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def _build_mlp(self, in_dim, mlp_dim, out_dim):
        return nn.Sequential(OrderedDict([
            ("layer1", nn.Linear(in_dim, mlp_dim)),
            ("bn1", nn.SyncBatchNorm(mlp_dim)),
            ("relu1", nn.ReLU(inplace=True)),
            ("layer2", nn.Linear(mlp_dim, mlp_dim)),
            ("bn2", nn.SyncBatchNorm(mlp_dim)),
            ("relu2", nn.ReLU(inplace=True)),
            ("layer3", nn.Linear(mlp_dim, out_dim)),
        ]))

    def encode_image(self, image):
        x = self.visual(image)

        return x

    def encode_text(self, text):
        x = self.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        x = x[torch.arange(x.shape[0]), text.argmax(dim=-1)]

        return x

    def forward(self, img_cat, label, text, *args, **kwargs):
        n_caption = img_cat.size(0)
        augs = torch.split(img_cat, 3, dim=1)
        n_views = len(augs)

        # do concat visual forward
        images = torch.cat(augs, dim=0)
        feats = self.visual(images)
        feats = torch.split(feats, n_caption, dim=0)

        # do separate BN
        res = [self.image_mlp(feat) for feat in feats]
        res_embedding = [feat @ self.image_projection
                         if self.vl_projection == 'linear'
                         else self.image_projection(feat)
                         for feat in feats]
        h = torch.cat(res, dim=0)
        h_emb = torch.cat(res_embedding, dim=0)

        label = label.view(-1, 1)
        label_expand = label.repeat(n_views, 1).squeeze()

        # text emb
        text_embed = self.encode_text(text)
        if self.vl_projection == 'linear':
            text_embed = text_embed @ self.text_projection
        else:
            text_embed = self.text_projection(text_embed)

        return {'image_feats': h,
                'image_emb': h_emb,
                'image_labels': label_expand,
                'text_emb': text_embed,
                'text_labels': label,
                'logit_scale': self.logit_scale.exp()}


def stablerep_mm_vit_small_patch16(ssl_mlp_dim, ssl_emb_dim, vl_projection, **kwargs):
    vision_model = timm.create_model('vit_small_patch16_224', num_classes=0)
    model = VisionTextEncoder(
        vision_width=384, vision_model=vision_model,
        ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim,
        context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12,
        vl_projection=vl_projection, embed_dim=512, **kwargs)
    return model


def stablerep_mm_vit_base_patch16(ssl_mlp_dim, ssl_emb_dim, vl_projection, **kwargs):
    vision_model = timm.create_model('vit_base_patch16_224', num_classes=0)
    model = VisionTextEncoder(
        vision_width=768, vision_model=vision_model,
        ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim,
        context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12,
        vl_projection=vl_projection, embed_dim=512, **kwargs)

    return model


def stablerep_mm_vit_large_patch16(ssl_mlp_dim, ssl_emb_dim, vl_projection, **kwargs):
    vision_model = timm.create_model('vit_large_patch16_224', num_classes=0)
    model = VisionTextEncoder(
        vision_width=1024, vision_model=vision_model,
        ssl_mlp_dim=ssl_mlp_dim, ssl_emb_dim=ssl_emb_dim,
        context_length=77, vocab_size=49408,
        transformer_width=512, transformer_heads=8, transformer_layers=12,
        vl_projection=vl_projection, embed_dim=512, **kwargs)

    return model


model_dict = {
    'small': stablerep_mm_vit_small_patch16,
    'base': stablerep_mm_vit_base_patch16,
    'large': stablerep_mm_vit_large_patch16,
}
