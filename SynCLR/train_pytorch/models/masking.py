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

import random
import math
from multiprocessing import Value
from abc import ABC
from logging import getLogger

import torch
import numpy as np


logger = getLogger()


class MaskingGenerator(ABC):
    def __init__(self, input_size):
        if not isinstance(input_size, tuple):
            input_size = (input_size,) * 2
        self.height, self.width = input_size
        self.num_patches = self.height * self.width

    def __repr__(self):
        raise NotImplementedError

    def get_shape(self):
        return self.height, self.width

    def _mask(self, mask, max_mask_patches):
        raise NotImplementedError

    def get_none_mask(self):
        return np.zeros(shape=self.get_shape(), dtype=bool)


class BlockMaskingGenerator(MaskingGenerator):
    def __init__(
        self,
        input_size,
        num_masking_patches=None,
        min_num_patches=4,
        max_num_patches=None,
        min_aspect=0.3,
        max_aspect=None,
    ):
        """
        Args:
            input_size: the size of the token map, e.g., 14x14
            num_masking_patches: how many masking patches
            min_num_patches: minimum number of patches per local masking block
            max_num_patches: maximum number of patches per local masking block
            min_aspect: min aspect ratio
            max_aspect: max aspect ratio
        """
        super().__init__(input_size)
        self.num_masking_patches = num_masking_patches
        self.min_num_patches = min_num_patches
        self.max_num_patches = num_masking_patches if max_num_patches is None else max_num_patches
        max_aspect = max_aspect or 1 / min_aspect
        self.log_aspect_ratio = (math.log(min_aspect), math.log(max_aspect))

    def __repr__(self):
        repr_str = f"Block Generator({self.height}, {self.width} -> [{self.min_num_patches} ~ " \
                   f"{self.max_num_patches}], max = {self.num_masking_patches,}, " \
                   f"{self.log_aspect_ratio[0]:.3f} ~ {self.log_aspect_ratio[1]:.3f})"
        return repr_str

    def _mask(self, mask, max_mask_patches):
        delta = 0
        for _ in range(10):
            target_area = random.uniform(self.min_num_patches, max_mask_patches)
            aspect_ratio = math.exp(random.uniform(*self.log_aspect_ratio))
            h = int(round(math.sqrt(target_area * aspect_ratio)))
            w = int(round(math.sqrt(target_area / aspect_ratio)))
            if w < self.width and h < self.height:
                top = random.randint(0, self.height - h)
                left = random.randint(0, self.width - w)

                num_masked = mask[top: top + h, left: left + w].sum()
                # overlap
                if 0 < h * w - num_masked <= max_mask_patches:
                    mask[top: top + h, left: left + w] = 1
                    delta = h * w - num_masked

                if delta > 0:
                    break
        return delta

    def __call__(self, num_masking_patches=0):
        mask = np.zeros(shape=self.get_shape(), dtype=bool)
        mask_count = 0
        while mask_count < num_masking_patches:
            max_mask_patches = num_masking_patches - mask_count
            max_mask_patches = min(max_mask_patches, self.max_num_patches)

            delta = self._mask(mask, max_mask_patches)
            if delta == 0:
                break
            else:
                mask_count += delta

        return mask


class Data2vecBlockMaskingGenerator(MaskingGenerator):

    def __init__(
        self,
        input_size,
        mask_prob: float = 0.75,
        mask_length: int = 3,
        mask_prob_adjust: float = 0.1,
        inverse_mask: bool = True,
        require_same_masks: bool = True,
        expand_adjcent: bool = False,
        mask_dropout: float = 0,
        non_overlapping: bool = False,
    ):
        super().__init__(input_size)
        self.mask_prob = mask_prob
        self.mask_length = mask_length
        self.mask_prob_adjust = mask_prob_adjust
        self.inverse_mask = inverse_mask
        self.require_same_masks = require_same_masks
        self.expand_adjcent = expand_adjcent
        self.mask_dropout = mask_dropout
        self.non_overlapping = non_overlapping

        assert self.mask_length > 1, "mask length must be greater than 1"

    def __repr__(self):
        repr_str = f"data2vec 2.0 Block Generator"
        return repr_str

    def _mask(self, mask, max_mask_patches):
        raise NotImplementedError

    @staticmethod
    def get_nbs(b, m, w):
        all_nbs = torch.nn.functional.conv2d(m.unsqueeze(1), w, padding="same")
        all_nbs = all_nbs.clamp_max_(1).view(b, -1)
        return all_nbs

    def __call__(self, b):
        """
            currently only supports batch generation
        """
        B, L = b, self.height * self.width

        mask_prob = self.mask_prob
        mask_length = self.mask_length
        mask_prob_adjust = self.mask_prob_adjust
        inverse_mask = self.inverse_mask
        require_same_masks = self.require_same_masks
        expand_adjcent = self.expand_adjcent
        mask_dropout = self.mask_dropout
        non_overlapping = self.non_overlapping

        d = int(L**0.5)

        if inverse_mask:
            mask_prob = 1 - mask_prob

        if non_overlapping:
            sz = math.ceil(d / mask_length)
            inp_len = sz * sz

            inp = torch.zeros((B, 1, sz, sz))
            w = torch.ones((1, 1, mask_length, mask_length))

            mask_inds = torch.multinomial(
                1 - inp.view(B, -1),
                int(inp_len * (mask_prob + mask_prob_adjust) * (1 + mask_dropout)),
                replacement=False,
            )
            inp.view(B, -1).scatter_(1, mask_inds, 1)

            mask = torch.nn.functional.conv_transpose2d(inp, w, stride=mask_length).squeeze(
                1
            )
            if mask.size(-1) > d:
                mask = mask[..., :d, :d]
        else:
            mask = torch.zeros((B, d, d))
            mask_inds = torch.randint(
                0,
                L,
                size=(
                    B,
                    int(
                        L
                        * ((mask_prob + mask_prob_adjust) / mask_length**2)
                        * (1 + mask_dropout)
                    ),
                ),
            )
            mask.view(B, -1).scatter_(1, mask_inds, 1)
            centers = mask.nonzero(as_tuple=True)

            inds = ([], [], [])

            offset = mask_length // 2
            for i in range(mask_length):
                for j in range(mask_length):
                    k1 = i - offset
                    k2 = j - offset
                    inds[0].append(centers[0])
                    inds[1].append(centers[1] + k1)
                    inds[2].append(centers[2] + k2)

            i0 = torch.cat(inds[0])
            i1 = torch.cat(inds[1]).clamp_(min=0, max=d - 1)
            i2 = torch.cat(inds[2]).clamp_(min=0, max=d - 1)

            mask[(i0, i1, i2)] = 1

        if require_same_masks and expand_adjcent:
            w = torch.zeros((1, 1, 3, 3))
            w[..., 0, 1] = 1
            w[..., 2, 1] = 1
            w[..., 1, 0] = 1
            w[..., 1, 2] = 1

            all_nbs = self.get_nbs(B, mask, w)

        mask = mask.reshape(B, -1)

        if require_same_masks:
            n_masks = mask.sum(dim=-1)
            final_target_len = int(L * (mask_prob))
            target_len = int(final_target_len * (1 + mask_dropout))

            for i in range(len(mask)):
                n = n_masks[i]
                m = mask[i]
                r = 0
                while expand_adjcent and n < target_len:
                    if r == 0:
                        nbs = all_nbs[i]
                    else:
                        nbs = self.get_nbs(1, m.view(1, d, d), w).flatten()

                    cands = (1 - m + nbs) > 1
                    cand_sz = int(cands.sum().item())

                    assert cand_sz > 0, f"{nbs} {cand_sz}"

                    to_mask = torch.multinomial(
                        cands.float(), min(cand_sz, int(target_len - n)), replacement=False
                    )
                    m[to_mask] = 1
                    assert to_mask.numel() > 0
                    n += to_mask.numel()
                    r += 1

                if n > final_target_len:
                    to_unmask = torch.multinomial(
                        m, int(n - final_target_len), replacement=False
                    )
                    m[to_unmask] = 0
                elif n < final_target_len:
                    to_mask = torch.multinomial(
                        (1 - m), int(final_target_len - n), replacement=False
                    )
                    m[to_mask] = 1

        if inverse_mask:
            mask = 1 - mask

        # to a list of bool tensor
        batch_masks = [m.bool().reshape(self.height, self.width) for m in mask]

        return batch_masks


class JepaBlockMaskingGenerator(MaskingGenerator):

    def __init__(
        self,
        input_size,
        enc_mask_scale=(0.85, 1.0),
        pred_mask_scale=(0.15, 0.2),
        aspect_ratio=(0.75, 1.5),
        nenc=1,
        npred=4,
        min_keep=10,
        allow_overlap=False,
        keep_shared_min=False,
        deterministic=True,
    ):
        """
        Args:
            input_size: the size of the token map, e.g., 14x14
            enc_mask_scale: the scale of the encoder masking block
            pred_mask_scale: the scale of the predictor masking block
            aspect_ratio: the aspect ratio of the masking block
            nenc: the number of encoder masking blocks
            npred: the number of predictor masking blocks
            min_keep: the minimum number of patches to keep
            allow_overlap: whether to allow overlap b/w enc and pred masks
            deterministic: if True, the masked size will be the same across GPUs, thus saving training time.
        """
        super().__init__(input_size)
        self.enc_mask_scale = enc_mask_scale
        self.pred_mask_scale = pred_mask_scale
        self.aspect_ratio = aspect_ratio
        self.nenc = nenc
        self.npred = npred
        self.min_keep = min_keep  # minimum number of patches to keep
        self.allow_overlap = allow_overlap  # whether to allow overlap b/w enc and pred masks
        self._itr_counter = Value('i', -1)  # collator is shared across worker processes
        self.keep_shared_min = keep_shared_min
        self.deterministic = deterministic

    def __repr__(self):
        repr_str = f"Jepa Block Generator"
        return repr_str

    def _mask(self, mask, max_mask_patches):
        raise NotImplementedError

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def _sample_block_size(self, generator, scale, aspect_ratio_scale):
        _rand = torch.rand(1, generator=generator).item()
        # -- Sample block scale
        min_s, max_s = scale
        mask_scale = min_s + _rand * (max_s - min_s)
        max_keep = int(self.height * self.width * mask_scale)
        # -- Sample block aspect-ratio
        min_ar, max_ar = aspect_ratio_scale
        aspect_ratio = min_ar + _rand * (max_ar - min_ar)
        # -- Compute block height and width (given scale and aspect-ratio)
        h = int(round(math.sqrt(max_keep * aspect_ratio)))
        w = int(round(math.sqrt(max_keep / aspect_ratio)))
        while h >= self.height:
            h -= 1
        while w >= self.width:
            w -= 1

        return h, w

    def _sample_block_mask(self, generator, b_size, acceptable_regions=None):
        h, w = b_size

        def constrain_mask(mask, tries=0):
            """ Helper to restrict given mask to a set of acceptable regions """
            N = max(int(len(acceptable_regions)-tries), 0)
            for k in range(N):
                mask *= acceptable_regions[k]
        # --
        # -- Loop to sample masks until we find a valid one
        tries = 0
        timeout = og_timeout = 20
        valid_mask = False
        while not valid_mask:
            # -- Sample block top-left corner
            if self.deterministic:
                top = torch.randint(0, self.height - h, (1,), generator=generator)
                left = torch.randint(0, self.width - w, (1,), generator=generator)
            else:
                top = torch.randint(0, self.height - h, (1,))
                left = torch.randint(0, self.width - w, (1,))
            mask = torch.zeros((self.height, self.width), dtype=torch.int32)
            mask[top:top+h, left:left+w] = 1
            # -- Constrain mask to a set of acceptable regions
            if acceptable_regions is not None:
                constrain_mask(mask, tries)
            mask = torch.nonzero(mask.flatten())
            # -- If mask too small try again
            valid_mask = len(mask) > self.min_keep
            if not valid_mask:
                timeout -= 1
                if timeout == 0:
                    tries += 1
                    timeout = og_timeout
                    logger.warning(f'Mask generator says: "Valid mask not found, decreasing acceptable-regions [{tries}]"')
        mask = mask.squeeze()
        # --
        mask_complement = torch.ones((self.height, self.width), dtype=torch.int32)
        mask_complement[top:top+h, left:left+w] = 0
        # --
        return mask, mask_complement

    def __call__(self, b):
        """
            currently only supports batch generation
        """
        B = b

        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)
        p_size = self._sample_block_size(
            generator=g,
            scale=self.pred_mask_scale,
            aspect_ratio_scale=self.aspect_ratio)
        e_size = self._sample_block_size(
            generator=g,
            scale=self.enc_mask_scale,
            aspect_ratio_scale=(1., 1.))

        collated_masks_pred, collated_masks_enc = [], []
        min_keep_pred = self.height * self.width
        min_keep_enc = self.height * self.width
        for _ in range(B):

            masks_p, masks_C = [], []
            for _ in range(self.npred):
                mask, mask_C = self._sample_block_mask(g, p_size)
                masks_p.append(mask)
                masks_C.append(mask_C)
                min_keep_pred = min(min_keep_pred, len(mask))
            collated_masks_pred.append(masks_p)

            acceptable_regions = masks_C
            try:
                if self.allow_overlap:
                    acceptable_regions= None
            except Exception as e:
                logger.warning(f'Encountered exception in mask-generator {e}')

            masks_e = []
            for _ in range(self.nenc):
                mask, _ = self._sample_block_mask(g, e_size, acceptable_regions=acceptable_regions)
                masks_e.append(mask)
                min_keep_enc = min(min_keep_enc, len(mask))
            collated_masks_enc.append(masks_e)

        if self.keep_shared_min:
            collated_masks_enc = [[cm[:min_keep_enc] for cm in cm_list] for cm_list in collated_masks_enc]
            collated_masks_pred = [[cm[:min_keep_pred] for cm in cm_list] for cm_list in collated_masks_pred]

        batch_masks = []
        for i in range(B):
            mask_indices = collated_masks_enc[i][0]

            flag = np.ones(shape=(self.height * self.width,), dtype=bool)
            flag[mask_indices] = False
            flag = flag.reshape(14, 14)
            batch_masks.append(torch.BoolTensor(flag))

        if self.keep_shared_min:
            collated_masks_enc = torch.utils.data.default_collate(collated_masks_enc)
            collated_masks_pred = torch.utils.data.default_collate(collated_masks_pred)
            collated_masks_enc = torch.stack(collated_masks_enc)
            collated_masks_pred = torch.stack(collated_masks_pred)

        return batch_masks, collated_masks_enc, collated_masks_pred


class RandomMaskingGenerator(MaskingGenerator):
    def __init__(
        self,
        input_size,
    ):
        """
        Args:
            input_size: the size of the token map, e.g., 14x14
        """
        super().__init__(input_size)

    def __repr__(self):
        repr_str = f"Random Generator({self.height}, {self.width})"
        return repr_str

    def _mask(self, mask, max_mask_patches):
        return super()._mask(mask, max_mask_patches)

    def __call__(self, num_masking_patches=0):
        if num_masking_patches <= 0:
            return np.zeros(shape=self.get_shape(), dtype=bool)

        mask = np.hstack([np.ones(num_masking_patches, dtype=bool),
                          np.zeros(self.num_patches - num_masking_patches, dtype=bool)])
        np.random.shuffle(mask)
        mask = mask.reshape(self.get_shape())
        return mask


if __name__ == '__main__':
    # mask_generator = BlockMaskingGenerator(
    #     input_size=(8, 8),
    #     max_num_patches=8,
    # )
    # mask_generator = RandomMaskingGenerator(
    #     input_size=(8, 8)
    # )
    # print(repr(mask_generator))
    # mask = mask_generator(8)
    # print(mask)
    # print(mask.sum())

    # mask_generator = Data2vecBlockMaskingGenerator(
    #     input_size=(14, 14),
    # )
    mask_generator = JepaBlockMaskingGenerator(
        input_size=(14, 14),
        keep_shared_min=True,
    )

    mask = mask_generator(8)
    print(type(mask))
    print(mask[0].shape)
    print(type(mask[0]))
    mask = torch.stack(mask)
    print(mask.shape)
    print(mask.sum(2).sum(1))

