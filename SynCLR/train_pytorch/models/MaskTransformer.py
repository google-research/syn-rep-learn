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

import math
import torch
import torch.nn as nn

from models.vision_transformer import VisionTransformer


def compute_gather_ids(masks):
    unmask_indices = masks.logical_not().nonzero(as_tuple=False)
    ids_keep = unmask_indices[:, -1].reshape(masks.shape[0], -1)
    return ids_keep


class MaskedTransformer(VisionTransformer):
    """Inherit vision transformer from timm"""

    def __init__(self, mask_style='ibot', **kwargs):
        super().__init__(**kwargs)
        assert mask_style in ["ibot", "mae", "none"], "mask_style must be `ibot`, `mae`, or `none`"

        self.patch_size = self.patch_embed.patch_size
        if isinstance(self.patch_size, tuple):
            self.patch_size = self.patch_size[0]

        nn.init.normal_(self.cls_token, std=1e-6)

        self.mask_style = mask_style
        if self.mask_style == "ibot":
            self.mask_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))
            torch.nn.init.normal_(self.mask_token, std=.02)

    def interpolate_pos_encoding(self, x, w, h, npatch):
        previous_dtype = x.dtype
        N = self.pos_embed.shape[1] - 1
        if npatch == N and w == h:
            return self.pos_embed
        pos_embed = self.pos_embed.float()
        class_pos_embed = pos_embed[:, 0]
        patch_pos_embed = pos_embed[:, 1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size
        h0 = h // self.patch_size
        # we add a small number to avoid floating point error in the interpolation
        # see discussion at https://github.com/facebookresearch/dino/issues/8
        w0, h0 = w0 + 0.1, h0 + 0.1

        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode="bicubic",
        )

        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1).to(previous_dtype)

    def prepare_tokens_with_masks(self, x, masks=None):
        """
        Args:
            x: data w/ shape [b, c, h, w]
            masks: shape [b, n], n is the number of tokens, 1 means masked, 0 means unmasked
        """
        b, c, h, w = x.shape
        x = self.patch_embed(x)
        if masks is not None:
            if self.mask_style == 'ibot':
                x = torch.where(masks.unsqueeze(-1), self.mask_token.to(x.dtype), x)
            elif self.mask_style == 'mae':  # only gather unmasked patches
                # add pos_embed before shuffle
                pos_embed = self.interpolate_pos_encoding(x, w, h, npatch=x.shape[1])
                x = x + pos_embed[:, 1:, :]
                ids_keep = compute_gather_ids(masks)
                x = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, x.shape[-1]))
                # x = x[masks.logical_not()]
                # x = x.reshape(b, -1, x.size(-1))
            else:
                raise NotImplementedError(f"mask style {self.mask_style} is not supported")

        if (masks is None) or (self.mask_style != "mae"):
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
            x = x + self.interpolate_pos_encoding(x, w, h, npatch=x.shape[1]-1)
        else:
            # mae-style masking, only need to add cls tokens w/ pos embedding
            cls_token = self.cls_token + self.pos_embed[:, :1, :]
            x = torch.cat((cls_token.expand(x.shape[0], -1, -1), x), dim=1)

        return x

    def forward_features_list(self, x_list, masks_list):
        x = [self.prepare_tokens_with_masks(x, masks) for x, masks in zip(x_list, masks_list)]

        num_data = len(x)
        if self.return_layer_targets:
            all_layer_results = [[] for _ in range(num_data)]
            for i, blk in enumerate(self.blocks):
                out = [blk(t) for t in x]
                x = [o[0] for o in out]
                # store layer targets
                for j in range(num_data):
                    all_layer_results[j].append(out[j][1])
            all_x = x
        else:
            all_x = [self.blocks(t) for t in x]
            all_layer_results = [None for _ in range(num_data)]

        output = []
        for x, masks, layer_results in zip(all_x, masks_list, all_layer_results):
            x_norm = self.norm(x)
            output.append(
                {
                    "x_norm": x_norm,
                    "x_norm_clstoken": x_norm[:, 0],
                    "x_norm_patchtokens": x_norm[:, 1:],
                    "masks": masks,
                    "layer_results": layer_results,
                }
            )
        return output

    def forward_features(self, x, masks=None):
        if isinstance(x, list):
            return self.forward_features_list(x, masks)

        x = self.prepare_tokens_with_masks(x, masks)

        if self.return_layer_targets:
            layer_results = []
            for i, blk in enumerate(self.blocks):
                x, lr = blk(x)
                layer_results.append(lr)
        else:
            x = self.blocks(x)
            layer_results = None

        x_norm = self.norm(x)
        return {
            "x_norm": x_norm,
            "x_norm_clstoken": x_norm[:, 0],
            "x_norm_patchtokens": x_norm[:, 1:],
            "masks": masks,
            "layer_results": layer_results,
        }

    def forward(self, *args, is_training=False, **kwargs):
        ret = self.forward_features(*args, **kwargs)
        if is_training:
            return ret
        else:
            return ret["x_norm_clstoken"]


def mask_vit_tiny(patch_size=16, **kwargs):
    model = MaskedTransformer(
        patch_size=patch_size, embed_dim=192, depth=12, num_heads=3, **kwargs)
    return model


def mask_vit_small(patch_size=16, **kwargs):
    model = MaskedTransformer(
        patch_size=patch_size, embed_dim=384, depth=12, num_heads=6, **kwargs)
    return model


def mask_vit_base(patch_size=16, **kwargs):
    model = MaskedTransformer(
        patch_size=patch_size, embed_dim=768, depth=12, num_heads=12, **kwargs)
    return model


def mask_vit_large(patch_size=16, **kwargs):
    model = MaskedTransformer(
        patch_size=patch_size, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    return model


def mask_vit_huge(patch_size=16, **kwargs):
    model = MaskedTransformer(
        patch_size=patch_size, embed_dim=1280, depth=32, num_heads=16, **kwargs)
    return model


model_dict = {
    'small': mask_vit_small,
    'base': mask_vit_base,
    'large': mask_vit_large,
}


def build_model(args):
    vit_kwargs = dict(
        num_classes=0,
        mask_style=args.mask_style,
        patch_size=args.patch_size
    )
    teacher = model_dict[args.model](**vit_kwargs,
                                     ffn_targets=args.ffn_targets,
                                     return_layer_targets=args.return_layer_targets)
    student = model_dict[args.model](
        **vit_kwargs,
        drop_path_rate=args.drop_path_rate,
    )
    embed_dim = student.embed_dim
    return student, teacher, embed_dim


if __name__ == '__main__':
    model = mask_vit_base(patch_size=16, num_classes=0, mask_style='mae')
    x = torch.randn(2, 3, 96, 96)
    out = model(x)
    print(out.shape)

