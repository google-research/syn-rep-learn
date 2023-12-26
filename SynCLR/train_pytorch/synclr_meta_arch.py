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
from torch import nn
import torch.nn.functional as F

from models.MaskTransformer import build_model
from models.head import build_contrast_head, build_dino_head
from models.mask_decoder import build_feature_predictor, Identity, apply_masks
from models.mask_losses import SupConMomentumLoss, DINOLoss, iBOTPatchLoss, MAEFeatLoss


class MetaArch(nn.Module):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        student_model_dict = dict()
        teacher_model_dict = dict()

        student, teacher, embed_dim = build_model(cfg)
        student_model_dict['backbone'] = student
        teacher_model_dict['backbone'] = teacher
        self.embed_dim = embed_dim

        # initialize parameters and checks
        self.total_n_global_crops = cfg.batch_size * cfg.n_img * cfg.global_crops_number
        self.total_n_local_crops = cfg.batch_size * cfg.n_img * cfg.local_crops_number
        # check flag conflicts
        if cfg.mask_style == "mae" and 0 < cfg.mask_probability < 1:
            assert cfg.mask_first_n, "need to mask first n sample to support `mae` style encoding"
        if cfg.clone_batch > 1 and cfg.mask_probability > 0:
            assert cfg.mask_style == "mae", "currently only `mae` style encoding supports clone batch"
        # setup feature decoder options
        self.do_feature_decoder = cfg.do_feature_decoder

        # add contrastive head, and loss
        self.do_contrast = cfg.contrast_loss_weight > 0
        if self.do_contrast:
            student_proj, student_pred, teacher_proj = build_contrast_head(cfg, embed_dim)
            self.contrast_loss = SupConMomentumLoss(
                temperature=cfg.contrast_temperature,
                avoid_local_global_same=cfg.avoid_local_global_same
            )
        else:
            student_proj = Identity()
            student_pred = Identity()
            teacher_proj = Identity()

        student_model_dict['proj'] = student_proj
        student_model_dict['pred'] = student_pred
        teacher_model_dict['proj'] = teacher_proj

        # add dino & ibot head, and loss
        self.do_dino = cfg.dino_loss_weight > 0 and cfg.local_crops_number > 0
        self.do_ibot = cfg.ibot_loss_weight > 0 and cfg.mask_probability > 0
        if self.do_ibot:

            assert cfg.mask_style == "ibot" or (cfg.mask_style == "mae" and cfg.mask_first_n), \
                "currently do_ibot needs `ibot` mask style or `mae` style with `mask_first_n`"
        if self.do_dino or self.do_ibot:
            student_dino_head, teacher_dino_head = build_dino_head(cfg, embed_dim)
            student_model_dict['dino_head'] = student_dino_head
            teacher_model_dict['dino_head'] = teacher_dino_head

        if self.do_dino:
            self.dino_loss = DINOLoss(cfg.dino_output_dim)
        if self.do_ibot:
            self.ibot_loss = iBOTPatchLoss(cfg.dino_output_dim)

        # add mae feature prediction head, and loss
        self.do_mae = cfg.mae_loss_weight > 0 and cfg.mask_probability > 0
        if self.do_mae:
            assert cfg.mask_style == "mae", "mae feature loss requires `mae` encoding style"
            assert cfg.mask_first_n, "need to mask first n sample to support `mae` style encoding"
            assert cfg.return_layer_targets, "need to return features from teacher model as the target for "
            self.do_feature_decoder = True
            self.mae_loss = MAEFeatLoss(cfg.regression_loss_scale, cfg.use_smooth_l1)

        if cfg.use_multi_pred:
            assert self.do_feature_decoder and self.do_mae, "multi-prediction requires feature decoder and mae loss"

        if self.do_feature_decoder:
            num_patches = (cfg.global_crops_size // cfg.patch_size) ** 2
            feature_decoder = build_feature_predictor(cfg, embed_dim, num_patches)
            student_model_dict['feature_decoder'] = feature_decoder
        else:
            student_model_dict['feature_decoder'] = Identity()

        self.student = nn.ModuleDict(student_model_dict)
        self.teacher = nn.ModuleDict(teacher_model_dict)

        # copy and  freeze the teacher
        to_copy_module = ['backbone', 'proj']
        if self.do_dino or self.do_ibot:
            to_copy_module.append('dino_head')
        self.momentum_module = to_copy_module

        for k in to_copy_module:
            self.teacher[k].load_state_dict(self.student[k].state_dict())
        for param in self.teacher.parameters():
            param.requires_grad = False

    def make_mae_feature_targets(self, y):
        target_layer_results = y[-self.cfg.average_top_k_layers:]
        if self.cfg.instance_norm_target_layer:
            target_layer_results = [
                tl.transpose(1, 2) for tl in target_layer_results  # BTC -> BCT
            ]
            target_layer_results = [
                F.instance_norm(tl.float()) for tl in target_layer_results
            ]
            target_layer_results = [
                tl.transpose(1, 2) for tl in target_layer_results  # BCT -> BTC
            ]
        y = target_layer_results[0].float()
        for tl in target_layer_results[1:]:
            y.add_(tl.float())
        y = y.div_(len(target_layer_results))

        if self.cfg.layer_norm_targets:
            y = F.layer_norm(y, y.shape[-1:])

        if self.cfg.instance_norm_targets:
            y = F.instance_norm(y.transpose(1, 2)).transpose(1, 2)

        return y

    def forward(self, inputs, teacher_temp=0.04):
        global_crops = inputs["collated_global_crops"]
        local_crops = inputs["collated_local_crops"]
        global_labels = inputs["collated_global_labels"]
        local_labels = inputs["collated_local_labels"]

        masks = inputs["collated_masks"]
        mask_indices_list = inputs["mask_indices_list"]
        masks_weight = inputs["masks_weight"]
        n_masked_patches = mask_indices_list.shape[0]
        n_masked_patches_tensor = inputs["n_masked_patches"]
        upperbound = inputs["upperbound"]

        masks_enc = inputs["masks_enc"]
        masks_pred = inputs["masks_pred"]

        n_global_crops = self.cfg.global_crops_number
        n_local_crops = self.cfg.local_crops_number

        # compute teacher output
        @torch.no_grad()
        def compute_teacher_output():
            teacher_backbone_output_dict = self.teacher.backbone(global_crops, is_training=True)
            teacher_cls_tokens = teacher_backbone_output_dict["x_norm_clstoken"]
            ibot_teacher_patch_tokens = teacher_backbone_output_dict["x_norm_patchtokens"]
            _dim = ibot_teacher_patch_tokens.shape[-1]
            n_cls_tokens = teacher_cls_tokens.shape[0]

            # get contrastive token
            teacher_contrast_cls = self.teacher.proj(teacher_cls_tokens)

            # dino and ibot
            if self.do_dino and not self.do_ibot:
                teacher_dino_tokens_after_head = self.teacher.dino_head(teacher_cls_tokens)
                teacher_ibot_patch_tokens_after_head = None
            elif not self.do_dino and self.do_ibot:
                teacher_dino_tokens_after_head = None
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound, _dim)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[:n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_ibot_patch_tokens_after_head = tokens_after_head[:n_masked_patches]
            elif self.do_dino and self.do_ibot:
                buffer_tensor_teacher = ibot_teacher_patch_tokens.new_zeros(upperbound + n_cls_tokens, _dim)
                buffer_tensor_teacher[:n_cls_tokens].copy_(teacher_cls_tokens)
                torch.index_select(
                    ibot_teacher_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list,
                    out=buffer_tensor_teacher[n_cls_tokens: n_cls_tokens + n_masked_patches],
                )
                tokens_after_head = self.teacher.dino_head(buffer_tensor_teacher)
                teacher_dino_tokens_after_head = tokens_after_head[:n_cls_tokens]
                teacher_ibot_patch_tokens_after_head = tokens_after_head[n_cls_tokens: n_cls_tokens + n_masked_patches]
            else:
                teacher_dino_tokens_after_head = None
                teacher_ibot_patch_tokens_after_head = None

            # normalize the output for dino/ibot loss
            if self.do_dino:
                if self.cfg.centering == "centering":
                    teacher_dino_softmaxed_centered_list = self.dino_loss.softmax_center_teacher(
                        teacher_dino_tokens_after_head, teacher_temp=teacher_temp
                    ).view(n_global_crops, -1, *teacher_dino_tokens_after_head.shape[1:])
                    self.dino_loss.update_center(teacher_dino_tokens_after_head)
                elif self.cfg.centering == "sinkhorn_knopp":
                    teacher_dino_softmaxed_centered_list = self.dino_loss.sinkhorn_knopp_teacher(
                        teacher_dino_tokens_after_head, teacher_temp=teacher_temp
                    ).view(n_global_crops, -1, *teacher_dino_tokens_after_head.shape[1:])
                else:
                    raise NotImplementedError
            else:
                teacher_dino_softmaxed_centered_list = None

            if self.do_ibot:
                if self.cfg.centering == "centering":
                    teacher_ibot_patch_tokens_after_head = teacher_ibot_patch_tokens_after_head.unsqueeze(0)
                    teacher_ibot_softmaxed_centered = self.ibot_loss.softmax_center_teacher(
                        teacher_ibot_patch_tokens_after_head[:, :n_masked_patches], teacher_temp=teacher_temp
                    )
                    teacher_ibot_softmaxed_centered = teacher_ibot_softmaxed_centered.squeeze(0)
                    self.ibot_loss.update_center(teacher_ibot_patch_tokens_after_head[:, :n_masked_patches])
                elif self.cfg.centering == "sinkhorn_knopp":
                    teacher_ibot_softmaxed_centered = self.ibot_loss.sinkhorn_knopp_teacher(
                        teacher_ibot_patch_tokens_after_head,
                        teacher_temp=teacher_temp,
                        n_masked_patches_tensor=n_masked_patches_tensor,
                    )
                else:
                    raise NotImplementedError
            else:
                teacher_ibot_softmaxed_centered = None

            # do mae feature loss
            if self.do_mae:
                if self.cfg.average_top_k_layers == -1 or self.cfg.use_multi_pred:
                    teacher_mae_targets = teacher_backbone_output_dict["x_norm"]
                    teacher_mae_targets = F.layer_norm(teacher_mae_targets, (teacher_mae_targets.size(-1),))
                else:
                    teacher_mae_layer_targets = teacher_backbone_output_dict["layer_results"]
                    teacher_mae_targets = self.make_mae_feature_targets(teacher_mae_layer_targets)
            else:
                teacher_mae_targets = None

            return teacher_contrast_cls, teacher_dino_softmaxed_centered_list, teacher_ibot_softmaxed_centered, \
                teacher_mae_targets

        # get the teacher outputs
        (
            teacher_contrast_cls,
            teacher_dino_softmaxed_centered_list,
            teacher_ibot_softmaxed_centered,
            teacher_mae_targets
        ) = compute_teacher_output()

        # process student global outputs
        n_masked_global_crops = int(self.total_n_global_crops * self.cfg.mask_probability)
        n_unmasked_global_crops = self.total_n_global_crops - n_masked_global_crops

        if self.cfg.mask_style == "mae" and self.cfg.mask_probability > 0:
            masked_student_global_backbone_output_dict = self.student.backbone(
                global_crops[:n_masked_global_crops].repeat_interleave(self.cfg.clone_batch, dim=0),
                masks[:n_masked_global_crops * self.cfg.clone_batch],
                is_training=True
            )

            if self.cfg.use_multi_pred:
                pred_x_norm = self.student.feature_decoder.forwar_multi_pred(
                    masked_student_global_backbone_output_dict["x_norm"],
                    masks_enc,
                    masks_pred
                )
            else:
                pred_x_norm = self.student.feature_decoder(
                    masked_student_global_backbone_output_dict["x_norm"],
                    masks[:n_masked_global_crops * self.cfg.clone_batch]
                )

            if n_unmasked_global_crops > 0:
                unmasked_student_global_backbone_output_dict = self.student.backbone(
                    global_crops[n_masked_global_crops:], is_training=True)
                # student_global_cls_token = torch.cat(
                #     [pred_x_norm[:, 0], unmasked_student_global_backbone_output_dict["x_norm_clstoken"]], dim=0)
                student_global_cls_token = torch.cat([
                    masked_student_global_backbone_output_dict["x_norm_clstoken"],
                    unmasked_student_global_backbone_output_dict["x_norm_clstoken"]
                ], dim=0)
                # w/ feature decoder, the dimension is matched, so we can concatenate
                if self.do_feature_decoder:
                    student_global_ibot_patch_tokens = torch.cat([
                        pred_x_norm[:, 1:],
                        unmasked_student_global_backbone_output_dict["x_norm_patchtokens"]
                    ], dim=0) if not self.cfg.use_multi_pred else pred_x_norm
                else:
                    student_global_ibot_patch_tokens = pred_x_norm[:, 1:]
            else:
                # student_global_cls_token = pred_x_norm[:, 0]
                student_global_cls_token = masked_student_global_backbone_output_dict["x_norm_clstoken"]
                student_global_ibot_patch_tokens = pred_x_norm[:, 1:] if not self.cfg.use_multi_pred else pred_x_norm
        else:
            cur_masks = masks if self.cfg.mask_probability > 0 else None
            student_global_backbone_output_dict = self.student.backbone(global_crops, cur_masks, is_training=True)
            student_global_cls_token = student_global_backbone_output_dict["x_norm_clstoken"]
            student_global_ibot_patch_tokens = student_global_backbone_output_dict["x_norm_patchtokens"]

        # process student local outputs
        if n_local_crops > 0:
            student_local_backbone_output_dict = self.student.backbone(local_crops, masks=None, is_training=True)
            student_local_cls_token = student_local_backbone_output_dict["x_norm_clstoken"]
        else:
            student_local_cls_token = local_crops   # empty tensor

        # contrastive loss
        if self.do_contrast:
            student_all_cls_token = torch.cat([student_global_cls_token, student_local_cls_token], dim=0)
            student_contrast_cls = self.student.pred(self.student.proj(student_all_cls_token))
            student_labels = torch.cat([
                torch.cat([
                    global_labels[:n_masked_global_crops].repeat_interleave(self.cfg.clone_batch),
                    global_labels[n_masked_global_crops:]
                ]),
                local_labels]
            )
            inputs_to_contrast_loss = {
                "student_contrast_cls": student_contrast_cls,
                "teacher_contrast_cls": teacher_contrast_cls,
                "student_labels": student_labels,
                "teacher_labels": global_labels
            }
            contrast_loss = self.contrast_loss(inputs_to_contrast_loss)
        else:
            contrast_loss = {
                'global_crop_contrast_loss': 0.0,
                'local_crop_contrast_loss': 0.0,
                'total_contrast_loss': 0.0,
            }

        # dino and ibot loss
        _dim = student_global_ibot_patch_tokens.shape[-1]
        if self.do_dino and not self.do_ibot:
            student_local_dino_tokens_after_head = self.student.dino_head(student_local_cls_token)
            student_global_ibot_tokens_after_head = None
        elif not self.do_dino and self.do_ibot:
            student_local_dino_tokens_after_head = None
            buffer_tensor_student = student_global_ibot_patch_tokens.new_zeros(upperbound, _dim)
            buffer_tensor_student[:n_masked_patches].copy_(
                torch.index_select(student_global_ibot_patch_tokens.flatten(0, 1),
                                   dim=0,
                                   index=mask_indices_list)
            )
            tokens_after_head = self.student.dino_head(buffer_tensor_student)
            student_global_ibot_tokens_after_head = tokens_after_head[:n_masked_patches]
        elif self.do_dino and self.do_ibot:
            n_local_cls_tokens = student_local_cls_token.shape[0]
            buffer_tensor_student = student_global_ibot_patch_tokens.new_zeros(n_local_cls_tokens + upperbound, _dim)
            buffer_tensor_student[:n_local_cls_tokens].copy_(student_local_cls_token)
            buffer_tensor_student[n_local_cls_tokens: n_local_cls_tokens + n_masked_patches].copy_(
                torch.index_select(student_global_ibot_patch_tokens.flatten(0, 1),
                                   dim=0,
                                   index=mask_indices_list,
                )
            )
            tokens_after_head = self.student.dino_head(buffer_tensor_student)
            student_local_dino_tokens_after_head = tokens_after_head[:n_local_cls_tokens]
            student_global_ibot_tokens_after_head = tokens_after_head[n_local_cls_tokens:
                                                                      n_local_cls_tokens + n_masked_patches]
        else:
            student_local_dino_tokens_after_head = student_global_ibot_tokens_after_head = None

        if self.do_dino:
            dino_local_loss = self.dino_loss(
                student_output_list=student_local_dino_tokens_after_head.chunk(n_local_crops),
                teacher_out_softmaxed_centered_list=teacher_dino_softmaxed_centered_list,
            ) / n_local_crops
        else:
            dino_local_loss = 0.0

        if self.do_ibot:
            ibot_loss = self.ibot_loss.forward_masked(
                student_global_ibot_tokens_after_head,
                teacher_ibot_softmaxed_centered,
                student_masks_flat=masks,
                n_masked_patches=n_masked_patches,
                masks_weight=masks_weight,
                n_masked_samples=n_masked_global_crops*self.cfg.clone_batch,
            ) / n_global_crops
        else:
            ibot_loss = 0.0

        # mae feature decoding loss
        if self.do_mae:
            if self.cfg.clone_batch > 1:
                with torch.no_grad():
                    teacher_mae_targets = torch.cat([
                        teacher_mae_targets[:n_masked_global_crops].repeat_interleave(self.cfg.clone_batch, dim=0),
                        teacher_mae_targets[n_masked_global_crops:]
                    ])

            if self.cfg.use_multi_pred:
                student_pred_tokens = student_global_ibot_patch_tokens
                teacher_target_tokens = apply_masks(teacher_mae_targets[:, 1:], masks_pred)
                mae_loss = F.smooth_l1_loss(student_pred_tokens, teacher_target_tokens)
            else:
                student_pred_tokens = torch.index_select(
                    student_global_ibot_patch_tokens.flatten(0, 1),
                    dim=0,
                    index=mask_indices_list)
                teacher_target_tokens = torch.index_select(
                    teacher_mae_targets[:, 1:].flatten(0, 1),  # remove the first cls token
                    dim=0,
                    index=mask_indices_list)
                mae_loss = self.mae_loss(
                    student_pred_tokens,
                    teacher_target_tokens,
                    student_masks_flat=masks,
                    n_masked_patches=n_masked_patches,
                    masks_weight=masks_weight,
                    n_masked_samples=n_masked_global_crops * self.cfg.clone_batch,
                ) / n_global_crops
        else:
            mae_loss = 0.0

        # compute the total loss
        total_loss = contrast_loss["total_contrast_loss"] * self.cfg.contrast_loss_weight + \
            dino_local_loss * self.cfg.dino_loss_weight + ibot_loss * self.cfg.ibot_loss_weight + \
            mae_loss * self.cfg.mae_loss_weight

        # return the final loss dict
        loss_dict = {**contrast_loss, "dino_loss": dino_local_loss, "ibot_loss": ibot_loss,
                     "mae_loss": mae_loss, "loss": total_loss}
        return loss_dict

    def update_teacher(self, m):
        """Momentum update of the teacher"""
        with torch.no_grad():
            for k in self.momentum_module:
                for param_b, param_m in zip(self.student[k].parameters(), self.teacher[k].parameters()):
                    param_m.data = param_m.data * m + param_b.data * (1. - m)


if __name__ == '__main__':

    import argparse
    parser = argparse.ArgumentParser('test ssl meta architecture', add_help=False)

    parser.add_argument('--model', default='small', type=str)
    parser.add_argument('--mask_style', default='mae', type=str)
    parser.add_argument('--drop_path_rate', default=0.0, type=float)

    parser.add_argument('--contrast_mlp_dim', default=512, type=int)
    parser.add_argument('--contrast_output_dim', default=128, type=int)
    parser.add_argument('--contrast_mlp_norm', default='ln', type=str)

    parser.add_argument('--contrast_temperature', default=0.1, type=float)
    parser.add_argument('--avoid_local_global_same', action='store_true')

    args = parser.parse_args()
    args.avoid_local_global_same = True
    model = MetaArch(args)

    inputs = dict()
    inputs["collated_global_crops"] = torch.randn(4, 3, 224, 224)
    inputs["collated_global_labels"] = torch.Tensor([1, 1, 2, 2])
    # inputs["collated_local_crops"] = torch.randn(12, 3, 96, 96)
    # inputs["collated_local_labels"] = torch.Tensor([1, 1, 2, 2]).repeat(3)
    inputs["collated_local_crops"] = torch.tensor([])
    inputs["collated_local_labels"] = torch.Tensor([1, 1, 2, 2]).repeat(0)
    loss_dict = model(inputs)
    for k, v in loss_dict.items():
        print(k, v.item())

