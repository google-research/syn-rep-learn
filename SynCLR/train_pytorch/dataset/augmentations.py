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

import logging
import random
from torchvision import transforms


logger = logging.getLogger("dinov2")


class GaussianBlur(transforms.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = transforms.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=p)


class MultiCropDataAugmentation(object):
    def __init__(
        self,
        global_crops_scale,
        local_crops_scale,
        global_crops_number,
        local_crops_number,
        global_crops_size=224,
        local_crops_size=96,
    ):
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.global_crops_number = global_crops_number
        self.local_crops_number = local_crops_number
        self.global_crops_size = global_crops_size
        self.local_crops_size = local_crops_size

        logger.info("###################################")
        logger.info("Using data augmentation parameters:")
        logger.info(f"global_crops_scale: {global_crops_scale}")
        logger.info(f"local_crops_scale: {local_crops_scale}")
        logger.info(f"global_crops_number: {global_crops_number}")
        logger.info(f"local_crops_number: {local_crops_number}")
        logger.info(f"global_crops_size: {global_crops_size}")
        logger.info(f"local_crops_size: {local_crops_size}")
        logger.info("###################################")

        # random resized crop and flip
        self.geometric_augmentation_global = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    global_crops_size, scale=global_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        self.geometric_augmentation_local = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    local_crops_size, scale=local_crops_scale, interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
            ]
        )

        # color distorsions / blurring
        color_jittering = transforms.Compose(
            [
                transforms.RandomApply(
                    [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )

        global_transfo1_extra = GaussianBlur(p=1.0)

        global_transfo2_extra = transforms.Compose(
            [
                GaussianBlur(p=0.1),
                transforms.RandomSolarize(threshold=128, p=0.2),
            ]
        )

        local_transfo_extra = GaussianBlur(p=0.5)

        # normalization
        self.normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        self.global_transfo1 = transforms.Compose([color_jittering, global_transfo1_extra, self.normalize])
        self.global_transfo2 = transforms.Compose([color_jittering, global_transfo2_extra, self.normalize])
        self.local_transfo = transforms.Compose([color_jittering, local_transfo_extra, self.normalize])

    def __call__(self, image):
        output = {"global_crops": [], "global_crops_teacher": []}
        seed = random.choice([0, 1])

        if isinstance(image, list):
            assert len(image) == self.global_crops_number + self.local_crops_number, \
                f"the length of the input list images doesn't match global+local numbers"

            # global crops
            for i in range(self.global_crops_number):
                im_base = self.geometric_augmentation_global(image[i])
                if (i + seed) % 2 == 0:
                    t = self.global_transfo1
                else:
                    t = self.global_transfo2
                crop = t(im_base)
                output["global_crops"].append(crop)
                output["global_crops_teacher"].append(crop)

            # local crops
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image[i]))
                for i in range(self.global_crops_number, self.global_crops_number + self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        else:
            # global crops
            for i in range(self.global_crops_number):
                im_base = self.geometric_augmentation_global(image)
                if (i + seed) % 2 == 0:
                    t = self.global_transfo1
                else:
                    t = self.global_transfo2
                crop = t(im_base)
                output["global_crops"].append(crop)
                output["global_crops_teacher"].append(crop)

            # local crops:
            local_crops = [
                self.local_transfo(self.geometric_augmentation_local(image)) for _ in range(self.local_crops_number)
            ]
            output["local_crops"] = local_crops
            output["offsets"] = ()

        return output


if __name__ == '__main__':
    # test 1
    import torch
    blur = GaussianBlur(p=1.0)
    print(blur.p)
    a = torch.rand(3, 32, 32)
    b = blur(a)
    print(b - a)

    # test 2
    from PIL import Image
    import numpy as np
    aug = MultiCropDataAugmentation(
        global_crops_scale=(0.32, 1.0),
        local_crops_scale=(0.08, 0.32),
        global_crops_number=2,
        local_crops_number=4,
    )
    img = torch.rand(3, 224, 224)
    img = img.numpy()
    img = (img * 255.).astype(np.uint8)
    img = img.transpose(1, 2, 0)
    img = Image.fromarray(img)
    out = aug(img)
    for k, v in out.items():
        print(k)
        if isinstance(v, list):
            for x in v:
                print(x.shape)
