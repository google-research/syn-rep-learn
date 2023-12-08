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

import os.path

import argparse
import json

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.datasets as datasets


from open_clip import create_model_and_transforms, get_tokenizer
from torch.nn import functional as F


def validate_zeroshot(val_loader, templates, labels, model, tokenizer, logit_scale):
    # switch to evaluate mode
    model.eval()
    total_top1 = 0
    total_images = 0
    total_loss = 0

    with torch.no_grad():
        text_features = []
        for label in labels:
            if isinstance(label, list):
                texts = [t.format(l) for t in templates for l in label]
            else:
                texts = [t.format(label) for t in templates]
            texts = tokenizer(texts).cuda(non_blocking=True)
            texts = texts.view(-1, 77).contiguous()
            class_embeddings = model.encode_text(texts)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            class_embeddings = class_embeddings.mean(dim=0)
            class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
            text_features.append(class_embeddings)
        text_features = torch.stack(text_features, dim=0)

        for images, target in val_loader:
            images = images.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

            # encode images
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # cosine similarity as logits
            logits_per_image = image_features @ text_features.t()

            # measure accuracy and record loss
            pred = logits_per_image.argmax(dim=1)
            correct = pred.eq(target).sum()
            loss = F.cross_entropy(logit_scale*logits_per_image, target, reduction='sum')

            total_top1 += correct.item()
            total_images += images.size(0)
            total_loss += loss.item()

    return 100 * total_top1 / total_images, total_loss / total_images


def main(args):
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        '',
        precision='amp',
        device='cuda',
        jit=False,
        force_quick_gelu=True,
        force_custom_text=False,
        force_patch_dropout=None,
        force_image_size=224,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        aug_cfg={},
        output_dict=True,
    )
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    state_dict = torch.load(args.ckpt, map_location=device)
    logit_scale = np.exp(state_dict['logit_scale'].item())
    msg = model.load_state_dict(state_dict, strict=True)
    print(msg)
    model = model.to(device)
    model.eval()
    print(f'evaluating {args.ckpt} ...')

    cudnn.benchmark = True

    with open('imagenet_labels.json') as f:
        labels = json.load(f)

    tokenizer = get_tokenizer(args.model)
    val_dataset = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=preprocess_val)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    templates = json.load(open('imagenet_templates.json'))
    acc, loss = validate_zeroshot(val_loader, templates, labels, model, tokenizer, logit_scale)
    print(f'ImageNet zero-shot accuracy: {acc}, loss: {loss}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP ImageNet zero-shot evaluation', add_help=False)
    parser.add_argument('--data-path', default='data/imagenet', type=str, help='path to imagenet dataset')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                        help='path to checkpoint to eval (default: none)')
    parser.add_argument('--batch-size', default=256, type=int, help='batch_size')
    parser.add_argument('--model', default='ViT-B-16', type=str, help='model architecture')
    parser.add_argument('-j', '--workers', default=10, type=int)
    args = parser.parse_args()
    main(args)
