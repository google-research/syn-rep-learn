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

import os
import PIL
import pandas as pd
from PIL import Image

from torchvision import datasets, transforms
from torch.utils.data import Dataset

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


class ImageNetCSVDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        image_path = self.data.iloc[index]["filename"]
        label = self.data.iloc[index]["label"]
        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.use_csv:
        if is_train:
            csv_path = '/mnt/yonglong_nfs_1/dataset/imagenet/train.csv'
        else:
            csv_path = '/mnt/yonglong_nfs_1/dataset/imagenet/val.csv'
        dataset = ImageNetCSVDataset(csv_path, transform=transform)
    else:
        root = os.path.join(args.data_path, 'train' if is_train else 'val')
        dataset = datasets.ImageFolder(root, transform=transform)

    print(dataset)

    return dataset


def build_transform(is_train, args):
    if args.inception_norm:
        mean = [0.5, 0.5, 0.5]
        std = [0.5, 0.5, 0.5]
    else:
        # mean = IMAGENET_DEFAULT_MEAN
        # std = IMAGENET_DEFAULT_STD
        # TODO: fix me
        mean = [0.48145466, 0.4578275, 0.40821073]
        std = [0.26862954, 0.26130258, 0.27577711]
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.crop_pct is None:
        if args.input_size <= 224:
            crop_pct = 224 / 256
        else:
            crop_pct = 1.0
    else:
        crop_pct = args.crop_pct
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
