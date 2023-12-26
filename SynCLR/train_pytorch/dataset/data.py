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
from PIL import Image
from torch.utils.data import Dataset
import csv
from tqdm import tqdm
csv.field_size_limit(500 * 1024 * 1024)


class SupconMultiCropDataset(Dataset):
    def __init__(self, input_filename, transforms=None, num_views=1, root_list=None):
        logging.debug(f'Loading csv data from {input_filename}.')
        self.images = []
        self.captions = []
        self.root_list = root_list
        self.num_views = num_views
        assert input_filename.endswith('.csv')
        with open(input_filename, 'r') as csv_file:
            csv_reader = csv.reader(csv_file)
            for row in tqdm(csv_reader):
                image = row[0]
                prompt = row[1]
                if image.endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(image)   # relative dir
                    self.captions.append(prompt)

        self.transforms = transforms
        logging.debug('Done loading data.')

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # sample #num_view image and transform them
        output_dict_list = []
        # random sample
        view_sample = np.random.choice(len(self.root_list), self.num_views, replace=False)
        for i in view_sample:
            image_path = os.path.join(self.root_list[i], self.images[idx])
            # open image and convert to rgb
            image = Image.open(image_path).convert('RGB')
            output_dict_list.append(self.transforms(image))

        # single image per caption
        if len(output_dict_list) == 1:
            return output_dict_list[0], idx

        # merge multiple image from the same caption
        output = {}
        for k, v in output_dict_list[0].items():
            output[k] = []
            for data in output_dict_list:
                if isinstance(v, list):
                    output[k].extend(data[k])
                else:
                    output[k].append(data[k])

        return output, idx
