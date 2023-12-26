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
import os.path

import torch.utils.data as data
from PIL import Image


def find_classes(file_path):
    with open(file_path, 'r') as f:
        classes = [d[3:] for d in f.read().splitlines()]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(path, root, class_to_idx):
    images = []
    labels = []
    for line in open(path, 'r'):
        line = os.path.join(root, line[1:].strip())
        assert os.path.isfile(line)
        images.append(line)
        for classname in class_to_idx:
            if f'/{classname}/' in line:
                labels.append(class_to_idx[classname])
                break

    return images, labels


class SUN397(data.Dataset):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=None):
        classes, class_to_idx = find_classes(os.path.join(root, 'ClassName.txt'))
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.train = train
        self.transform = transform
        self.target_transform = target_transform
        self.split = '01'

        train_path = os.path.join(root, 'Training_' + self.split + '.txt')
        test_path = os.path.join(root, 'Testing_' + self.split + '.txt')
        path = train_path if train else test_path

        self.images, self.labels = make_dataset(path, root, class_to_idx)
        assert (len(self.images) == len(self.labels))

    def __getitem__(self, index):
        _img = Image.open(self.images[index]).convert('RGB')
        _label = self.labels[index]
        if self.transform is not None:
            _img = self.transform(_img)
        if self.target_transform is not None:
            _label = self.target_transform(_label)

        return _img, _label

    def __len__(self):
        return len(self.images)
