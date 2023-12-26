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

from functools import partial

import torch
from torch import nn
from timm.models.vision_transformer import Block
from timm.models.layers import trunc_normal_

from util.pos_embed import get_2d_sincos_pos_embed


def apply_masks(x, masks):
    """
    :param x: tensor of shape [B (batch-size), N (num-patches), D (feature-dim)]
    :param masks: list of tensors containing indices of patches in [N] to keep
    """
    all_x = []
    for m in masks:
        mask_keep = m.unsqueeze(-1).repeat(1, 1, x.size(-1))
        all_x += [torch.gather(x, dim=1, index=mask_keep)]
    return torch.cat(all_x, dim=0)


def get_recover_ids(masks):
    mask_indices = masks.nonzero(as_tuple=False)
    unmask_indices = masks.logical_not().nonzero(as_tuple=False)
    unused = mask_indices[:, -1].reshape(masks.shape[0], -1)
    used = unmask_indices[:, -1].reshape(masks.shape[0], -1)
    all_indices = torch.cat([used, unused], dim=1)
    recover_ids = torch.argsort(all_indices, dim=1)
    return recover_ids


class Identity(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, input, *args, **kwargs):
        return input


class TransformerDecoder(nn.Module):
    def __init__(
        self,
        num_patches=196,
        embed_dim=768,
        decoder_embed_dim=384,
        decoder_depth=6,
        decoder_num_heads=12,
        mlp_ratio=4.0,
        fix_sin_cos_pos_embedding=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, decoder_embed_dim))

        self.decoder_blocks = nn.ModuleList([
            Block(decoder_embed_dim, decoder_num_heads, mlp_ratio, qkv_bias=True,
                  norm_layer=norm_layer)
            for i in range(decoder_depth)])

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(decoder_embed_dim, embed_dim, bias=True)

        self.num_patches = num_patches
        self.fix_sin_cos_pos_embedding = fix_sin_cos_pos_embedding
        self.initialize_weights()

    def initialize_weights(self):
        if self.fix_sin_cos_pos_embedding:  # fix the sin/cos pos embedding
            self.decoder_pos_embed.requires_grad = False
            decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                        int(self.num_patches ** .5),
                                                        cls_token=True)
            self.decoder_pos_embed.data.copy_(
                torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forwar_multi_pred(self, x, masks_x, masks):
        assert (masks is not None) and (masks_x is not None), 'Cannot run predictor without mask indices'

        x = self.decoder_embed(x)

        # add positional embedding
        x_pos_embed = self.decoder_pos_embed[:, 1:, :].repeat(x.size(0), 1, 1)
        x_pos_embed = apply_masks(x_pos_embed, masks_x)
        cls_pos_embed = self.decoder_pos_embed[:, :1, :].repeat(x_pos_embed.shape[0], 1, 1)
        x += torch.cat([cls_pos_embed, x_pos_embed], dim=1)

        _, n_context, D = x.shape

        pos_embed = self.decoder_pos_embed[:, 1:, :].repeat(x.size(0), 1, 1)
        pos_embed = apply_masks(pos_embed, masks)
        pred_tokens = self.mask_token.repeat(pos_embed.size(0), pos_embed.size(1), 1)
        pred_tokens += pos_embed
        x = x.repeat(len(masks), 1, 1)
        x = torch.cat([x, pred_tokens], dim=1)

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        x = x[:, n_context:]

        # predictor projection
        x = self.decoder_pred(x)

        return x

    def forward(self, x, masks=None):
        x = self.decoder_embed(x)

        if masks is not None:
            # x contains cls_token, so self.num_patchs + 1
            mask_tokens = self.mask_token.repeat(
                x.shape[0], self.num_patches + 1 - x.shape[1], 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token

            recover_ids = get_recover_ids(masks)
            x_ = torch.gather(x_, dim=1, index=recover_ids.unsqueeze(-1).repeat(1, 1, x.shape[2]))
            x = torch.cat([x[:, :1, :], x_], dim=1)  # prepend cls token

        # add pos embed
        x = x + self.decoder_pos_embed

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        return x


class SamePad2d(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.remove = 1 if kernel_size % 2 == 0 else 0

    def forward(self, x):
        assert len(x.size()) == 4
        if self.remove > 0:
            x = x[:, :, : -self.remove, : -self.remove]
        return x


class TransposeLast(nn.Module):
    def __init__(self, deconstruct_idx=None, tranpose_dim=-2):
        super().__init__()
        self.deconstruct_idx = deconstruct_idx
        self.tranpose_dim = tranpose_dim

    def forward(self, x):
        if self.deconstruct_idx is not None:
            x = x[self.deconstruct_idx]
        return x.transpose(self.tranpose_dim, -1)


class ConvolutionDecoder(nn.Module):
    """docstring for ConvolutionDecoder"""
    def __init__(
        self,
        embed_dim=768,
        h_size=14,
        w_size=14,
        decoder_dim=768,
        decoder_kernel=3,
        decoder_groups=16,
        decoder_layers=6,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.h_size = h_size
        self.w_size = w_size
        self.decoder_dim = decoder_dim
        self.decoder_kernel = decoder_kernel
        self.decoder_groups = decoder_groups
        self.decoder_layers = decoder_layers

        def make_block(in_dim):
            block = [
                nn.Conv2d(
                    in_dim,
                    decoder_dim,
                    kernel_size=decoder_kernel,
                    padding=decoder_kernel // 2,
                    groups=decoder_groups,
                ),
                SamePad2d(decoder_kernel),
                TransposeLast(tranpose_dim=-3),
                nn.LayerNorm(decoder_dim, elementwise_affine=False),
                TransposeLast(tranpose_dim=-3),
                nn.GELU(),
            ]

            return nn.Sequential(*block)

        self.blocks = nn.Sequential(
            *[
                make_block(embed_dim if i == 0 else decoder_dim)
                for i in range(decoder_layers)
            ]
        )

        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.proj = nn.Linear(decoder_dim, embed_dim)

    def reset_parameters(self):
        for mod in self.proj.modules():
            if isinstance(mod, nn.Linear):
                mod.reset_parameters()

    def forward(self, x, masks=None):
        B, T, C = x.shape

        # save cls tokens for further return
        cls_tokens = x[:, :1, :]

        if masks is not None:
            # x contains cls token, need extra +1
            mask_tokens = self.mask_token.repeat(B, self.h_size * self.w_size + 1 - T, 1)
            x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
            recover_ids = get_recover_ids(masks)
            x_ = torch.gather(x_, dim=1, index=recover_ids.unsqueeze(-1).repeat(1, 1, x.shape[2]))
        else:
            x_ = x[:, 1:, :]

        # transposed for CNN
        x = x_.transpose(1, 2).reshape(B, C, self.h_size, self.w_size)

        residual = x
        for i, layer in enumerate(self.blocks):
            x = layer(x)
            x = x + residual
            residual = x

        x = x.reshape(B, self.decoder_dim, self.h_size * self.w_size).transpose(1, 2)
        x = self.proj(x)

        return torch.cat([cls_tokens, x], dim=1)


def build_feature_predictor(args, embed_dim, num_patches):
    assert args.decoder_type in ["transformer", "cnn"], "only supports transformer and cnn decoder"
    if args.decoder_type == "transformer":
        decoder = TransformerDecoder(
            num_patches=num_patches,
            embed_dim=embed_dim,
            decoder_embed_dim=args.decoder_embed_dim,
            decoder_depth=args.decoder_depth,
            decoder_num_heads=args.decoder_num_heads,
        )
    elif args.decoder_type == "cnn":
        decoder = ConvolutionDecoder(
            embed_dim=embed_dim,
            h_size=int(num_patches ** 0.5),
            w_size=int(num_patches ** 0.5),
            decoder_dim=args.decoder_embed_dim,
            decoder_kernel=args.decoder_kernel,
            decoder_layers=args.decoder_depth,
        )
    else:
        raise NotImplementedError
    return decoder


if __name__ == '__main__':
    # === test TransformerDecoder
    # decoder = TransformerDecoder(
    #     num_patches=9,
    #     embed_dim=2,
    #     decoder_embed_dim=2,
    #     decoder_depth=1,
    #     decoder_num_heads=1,
    #     mlp_ratio=1.0,
    #     fix_sin_cos_pos_embedding=False
    # )
    #
    # from models.masking import RandomMaskingGenerator
    # masking_generator = RandomMaskingGenerator(
    #     input_size=(3,3)
    # )
    #
    # batch_masks = [masking_generator(num_masking_patches=6) for i in range(2)]
    # batch_masks = [torch.BoolTensor(m) for m in batch_masks]
    # masks = torch.stack(batch_masks, dim=0).flatten(1)
    #
    # tokens = torch.randn(2, 9, 2)
    # print(tokens)
    # print(masks.logical_not())
    # out = tokens[masks.logical_not()]
    # out = out.reshape(2, 3, 2)
    # cls_token = torch.zeros(2, 1, 2)
    #
    # x = torch.cat([cls_token, out], dim=1)
    # # print(x)
    # out = decoder(x, masks)
    # print(out)

    # === test ConvolutionDecoder
    decoder = ConvolutionDecoder(
        embed_dim=128,
        h_size=14,
        w_size=14,
        decoder_dim=128,
        decoder_kernel=3,
        decoder_groups=8,
        decoder_layers=6,
    )

    x = torch.randn(2, 196, 128)
    out = decoder(x)
    print(out.shape)
