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
from torch.nn.init import trunc_normal_
from torch.nn.utils import weight_norm


def _build_dino_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=None, use_bn=False, bias=True):
    if nlayers == 1:
        return nn.Linear(in_dim, bottleneck_dim, bias=bias)
    else:
        layers = [nn.Linear(in_dim, hidden_dim, bias=bias)]
        if use_bn:
            layers.append(nn.BatchNorm1d(hidden_dim))
        layers.append(nn.GELU())
        for _ in range(nlayers - 2):
            layers.append(nn.Linear(hidden_dim, hidden_dim, bias=bias))
            if use_bn:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_dim, bottleneck_dim, bias=bias))
        return nn.Sequential(*layers)


def _build_mlp_bn(num_layers, input_dim, mlp_dim, output_dim, last_norm=True):
    """mlp with batchnorm"""
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        mlp.append(nn.Linear(dim1, dim2, bias=False))
        # use SyncBatchNorm, no need to convert again
        if l < num_layers - 1:
            mlp.append(nn.SyncBatchNorm(dim2))
            mlp.append(nn.ReLU(inplace=True))
        elif last_norm:
            # follow SimCLR's design: https://github.com/google-research/simclr/blob/master/model_util.py#L157
            # for simplicity, we further removed gamma in BN
            mlp.append(nn.SyncBatchNorm(dim2, affine=False))

    return nn.Sequential(*mlp)


def _build_mlp_ln(num_layers, input_dim, mlp_dim, output_dim, last_norm=True):
    """mlp with layernorm"""
    mlp = []
    for l in range(num_layers):
        dim1 = input_dim if l == 0 else mlp_dim
        dim2 = output_dim if l == num_layers - 1 else mlp_dim

        if l < num_layers - 1:
            mlp.append(nn.Linear(dim1, dim2, bias=True))
            mlp.append(nn.LayerNorm(dim2))
            mlp.append(nn.GELU())
        else:
            mlp.append(nn.Linear(dim1, dim2, bias=False))
            if last_norm:
                mlp.append(nn.LayerNorm(dim2, elementwise_affine=False))

    return nn.Sequential(*mlp)


class DINOHead(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        use_bn=False,
        nlayers=3,
        hidden_dim=2048,
        bottleneck_dim=256,
        mlp_bias=True,
    ):
        super().__init__()
        nlayers = max(nlayers, 1)
        self.mlp = _build_dino_mlp(nlayers, in_dim, bottleneck_dim, hidden_dim=hidden_dim, use_bn=use_bn, bias=mlp_bias)
        self.apply(self._init_weights)
        self.last_layer = weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        eps = 1e-6 if x.dtype == torch.float16 else 1e-12
        x = nn.functional.normalize(x, dim=-1, p=2, eps=eps)
        x = self.last_layer(x)
        return x


class MLPHead(nn.Module):
    def __init__(
        self,
        num_layers,
        input_dim,
        mlp_dim,
        output_dim,
        mlp_norm='ln',
        last_norm=True
    ):
        super().__init__()

        assert mlp_norm in ["ln", "LN", "bn", "BN"], f"mlp norm type {mlp_norm} is not supported"
        self.mlp_norm = mlp_norm

        if self.mlp_norm == "bn" or self.mlp_norm == "BN":
            self.mlp = _build_mlp_bn(num_layers, input_dim, mlp_dim, output_dim, last_norm)
        elif self.mlp_norm == "ln" or self.mlp_norm == "LN":
            self.mlp = _build_mlp_ln(num_layers, input_dim, mlp_dim, output_dim, last_norm)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.mlp(x)


def build_contrast_head(args, embed_dim):
    proj_kwargs = dict(
        num_layers=3,
        input_dim=embed_dim,
        mlp_dim=args.contrast_mlp_dim,
        output_dim=args.contrast_output_dim,
        mlp_norm=args.contrast_mlp_norm,
    )
    pred_kwargs = dict(
        num_layers=2,
        input_dim=args.contrast_output_dim,
        mlp_dim=args.contrast_mlp_dim,
        output_dim=args.contrast_output_dim,
        mlp_norm=args.contrast_mlp_norm,
    )
    student_proj = MLPHead(**proj_kwargs)
    teacher_proj = MLPHead(**proj_kwargs)
    student_pred = MLPHead(**pred_kwargs)
    return student_proj, student_pred, teacher_proj


def build_dino_head(args, embed_dim):
    dino_kwargs = dict(
        in_dim=embed_dim,
        out_dim=args.dino_output_dim,
        nlayers=args.dino_n_layers,
        hidden_dim=args.dino_mlp_dim,
        bottleneck_dim=args.dino_bottleneck_dim,
    )
    student_dino_head = DINOHead(**dino_kwargs)
    teacher_dino_head = DINOHead(**dino_kwargs)
    return student_dino_head, teacher_dino_head


if __name__ == '__main__':
    # unit test
    # mlp = DINOHead(
    #     in_dim=128,
    #     out_dim=128,
    #     use_bn=False,
    #     nlayers=3,
    #     hidden_dim=2048,
    #     bottleneck_dim=256,
    #     mlp_bias=True,
    # )
    mlp = MLPHead(
        num_layers=3,
        input_dim=128,
        mlp_dim=2048,
        output_dim=128,
        mlp_norm='ln',
        last_norm=True
    )
    data = torch.randn(2, 128)
    out = mlp(data)
    print(out.shape)
