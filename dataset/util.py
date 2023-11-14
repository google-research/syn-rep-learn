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
from PIL import ImageFilter, ImageOps
import numpy as np
from typing import Union
import torchvision.transforms as transforms


class DownSampleAndUpsample(object):
    """downsample and then upsample images. This augmentation is applied
    to synthetic images which are all in high resolution. W/ this augmentation,
    we can narrow down the performance gap caused by resolution differences"""

    def __init__(self,
                 down_res: Union[list, int] = [128, 64],
                 p: list = None,
                 up_size: int = 256):
        # sanity check
        self.single_res = True
        if isinstance(down_res, list):
            if not p:
                p = [1 for _ in range(len(down_res))]
            assert len(p) == len(down_res), "lengths of down_res and p not matched"
            p = np.asarray(p)
            p = p / p.sum()
            self.single_res = False

        self.down_res = down_res
        self.p = p
        self.up_size = up_size

    def __call__(self, x):
        if self.single_res:
            res = self.down_res
        else:
            res = np.random.choice(self.down_res, p=self.p)
            res = int(res)

        x = transforms.Resize(res)(x)
        x = transforms.Resize(self.up_size)(x)
        return x


class GaussianBlur(object):
    """Gaussian blur https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[.1, 2.]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
