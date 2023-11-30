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
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import csv
import clip
from tqdm import tqdm
csv.field_size_limit(500 * 1024 * 1024)


class NCropTransform:
    """Create N crops from the same image"""
    def __init__(self, transform, n):
        self.transform = transform
        self.n = n

    def __call__(self, x):
        res = []
        for i in range(self.n):
            res.append(self.transform(x))
        return torch.cat(res, dim=0)


class SupconDataset(Dataset):
    def __init__(self, input_filename, transforms, num_views=1,
                 root_list=None, num_crop=1, tokenizer=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        assert num_crop >= 1, f'number of crops is less than 1: {num_crop}'
        self.images = []
        self.captions = []
        self.root_list = root_list
        self.num_views = num_views
        self.tokenizer = tokenizer
        assert input_filename.endswith('.csv')
        with open(input_filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in tqdm(csv_reader):
                image = row[0]
                prompt = row[1]
                if image.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(image)  # relative dir
                    self.captions.append(prompt)

        if num_crop > 1:
            self.transforms = NCropTransform(transforms, num_crop)
        else:
            self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # sample num_view of images and transform them
        images = []
        # random sample
        view_sample = np.random.choice(len(self.root_list),
                                       self.num_views,
                                       replace=False)
        for i in view_sample:
            image_path = os.path.join(self.root_list[i], self.images[idx])
            # open image and convert to rgb
            image = Image.open(image_path).convert('RGB')
            images.append(self.transforms(image))
        # concat image on channel dim
        images = torch.cat(images, dim=0)
        if self.tokenizer is None:
            return images, idx
        else:
            # texts = self.tokenizer(str(self.captions[idx]))
            texts = clip.tokenize(self.captions[idx], truncate=True).squeeze().long()
            return images, texts, idx
