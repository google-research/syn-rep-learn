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

import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast
from contextlib import nullcontext
from imwatermark import WatermarkEncoder
from torch.utils.data import DataLoader, Dataset

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

torch.set_grad_enabled(False)


class PromptDataset(Dataset):
    """Build prompt loading dataset"""

    def __init__(self, file, start, n_skip, outdir, opt):
        self.file = file
        self.n_skip = n_skip

        prompts = []
        with open(file, "r") as f:
            for line in f:
                line = line.strip()
                line = line.replace('\t', ' ')
                line = line.replace('\r', ' ')
                line = line.replace('\n', ' ')
                prompts.append(line)
        ids = np.arange(len(prompts))

        print(f"total prompts: {len(prompts)}")

        n_prompts_per_gpu = len(prompts) // n_skip + 1
        if start == n_skip - 1:
            self.prompts = prompts[n_prompts_per_gpu * start:]
            self.ids = ids[n_prompts_per_gpu * start:]
        else:
            self.prompts = prompts[n_prompts_per_gpu * start: n_prompts_per_gpu * (start + 1)]
            self.ids = ids[n_prompts_per_gpu * start: n_prompts_per_gpu * (start + 1)]

        # skip what has been generated, for resuming purpose
        self.outdir = outdir
        cur_id = self.skip_ids(opt)
        print(f"skipping {cur_id} images!")

        self.prompts = self.prompts[cur_id:]
        self.ids = self.ids[cur_id:]

        print(f"remained prompts: {len(prompts)}")

        self.num = len(self.prompts)

    def skip_ids(self, opt):

        if opt.split_size_folder > 0 and opt.split_size_image > 0:
            split_size_folder = opt.split_size_folder
            split_size_image = opt.split_size_image
        else:
            split_size_folder = opt.split_size
            split_size_image = opt.split_size

        cur_id = 0
        for i, id in enumerate(self.ids):
            folder_level_1 = id // (split_size_folder * split_size_image)
            folder_level_2 = (id - folder_level_1 * split_size_folder * split_size_image) // split_size_image
            image_id = id - folder_level_1 * split_size_folder * split_size_image - folder_level_2 * split_size_image
            file = os.path.join(self.outdir, f"{folder_level_1:06}", f"{folder_level_2:06}", f"{image_id:05}.png")
            if not os.path.isfile(file):
                break
            cur_id += 1
        return max(0, cur_id - 2)

    def __len__(self):
        return self.num

    def __getitem__(self, item):
        prompt = self.prompts[item]
        id = self.ids[item]

        return prompt, id


def load_model_from_config(config, ckpt, device=torch.device("cuda"), verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    if device == torch.device("cuda"):
        model.cuda()
    elif device == torch.device("cpu"):
        model.cpu()
        model.cond_stage_model.device = "cpu"
    else:
        raise ValueError(f"Incorrect device name. Received: {device}")
    model.eval()
    return model


class ImageSaver(object):

    def __init__(self, outdir, opt):

        if opt.split:
            assert (opt.split_size > 0) or (opt.split_size_folder > 0 and opt.split_size_image > 0), \
                'splitting parameter wrong'

        self.outdir = outdir
        self.split = opt.split
        if opt.split_size_folder > 0 and opt.split_size_image > 0:
            self.split_size_folder = opt.split_size_folder
            self.split_size_image = opt.split_size_image
        else:
            self.split_size_folder = opt.split_size
            self.split_size_image = opt.split_size
        self.save_size = opt.img_save_size
        self.last_folder_level_1 = -1
        self.last_folder_level_2 = -1
        os.makedirs(self.outdir, exist_ok=True)

        if self.split:
            self.cur_folder = None
        else:
            self.cur_folder = self.outdir

    def save(self, img, id):
        id = int(id)
        if self.split:
            # compute folder id and image id
            folder_level_1 = id // (self.split_size_folder * self.split_size_image)
            folder_level_2 = (id - folder_level_1 * self.split_size_folder * self.split_size_image) // self.split_size_image
            image_id = id - folder_level_1 * self.split_size_folder * self.split_size_image - folder_level_2 * self.split_size_image
            if (self.cur_folder is None) or (self.last_folder_level_1 != folder_level_1) or \
                    (self.last_folder_level_2 != folder_level_2):
                self.cur_folder = os.path.join(self.outdir, f"{folder_level_1:06}", f"{folder_level_2:06}")
                os.makedirs(self.cur_folder, exist_ok=True)
            self.last_folder_level_1 = folder_level_1
            self.last_folder_level_2 = folder_level_2
        else:
            image_id = id

        img.save(os.path.join(self.cur_folder, f"{image_id:05}.png"))


class StableGenerator(object):

    def __init__(self, model, opt):
        self.opt = opt
        # model
        self.model = model

        device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
        if opt.plms:
            sampler = PLMSSampler(model, device=device)
        elif opt.dpm:
            sampler = DPMSolverSampler(model, device=device)
        else:
            sampler = DDIMSampler(model, device=device)
        self.sampler = sampler

        # unconditional vector
        self.uc = model.get_learned_conditioning([""])
        if self.uc.ndim == 2:
            self.uc = self.uc.unsqueeze(0)
        self.batch_uc = None

        # shape
        self.shape = [opt.C, opt.H // opt.f, opt.W // opt.f]

        # precision scope
        self.precision_scope = autocast if opt.precision == "autocast" or opt.bf16 else nullcontext

    def generate(self, prompts, n_sample_per_prompt):
        with torch.no_grad():
            with self.precision_scope("cuda"):
                with self.model.ema_scope():

                    # prepare the unconditional vector
                    bsz = len(prompts) * n_sample_per_prompt
                    if self.batch_uc is None or self.batch_uc.shape[0] != bsz:
                        self.batch_uc = self.uc.expand(bsz, -1, -1)

                    # prepare the conditional vector
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = self.model.get_learned_conditioning(prompts)
                    batch_c = c.unsqueeze(1).expand(-1, n_sample_per_prompt, -1, -1)
                    batch_c = batch_c.reshape(bsz, batch_c.shape[-2], batch_c.shape[-1])

                    # sampling
                    samples_ddim, _ = self.sampler.sample(S=self.opt.steps,
                                                          conditioning=batch_c,
                                                          batch_size=bsz,
                                                          shape=self.shape,
                                                          verbose=False,
                                                          unconditional_guidance_scale=self.opt.scale,
                                                          unconditional_conditioning=self.batch_uc,
                                                          eta=self.opt.ddim_eta,
                                                          x_T=None)     # no fixed start code

                    x_samples_ddim = self.model.decode_first_stage(samples_ddim)
                    x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0, min=0.0, max=1.0)
                    x_samples_ddim = x_samples_ddim.cpu().permute(0, 2, 3, 1).numpy()
                    # x_samples_ddim = x_samples_ddim.cpu().numpy()
                    x_samples_ddim = 255. * x_samples_ddim

                    return x_samples_ddim


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--img_save_size",
        type=int,
        default=256,
        help="image saving size"
    )
    parser.add_argument(
        "--split",
        action='store_true',
        help="whether we split the data during saving (might further improve for many millions of images",
    )
    parser.add_argument(
        "--split_size",
        type=int,
        default=1000,
        help="split size for saving images"
    )
    parser.add_argument(
        "--split_size_folder",
        type=int,
        default=1000,
        help="split size for number of folders inside each first level folder"
    )
    parser.add_argument(
        "--split_size_image",
        type=int,
        default=1000,
        help="split size for number of images inside each second level folder"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--dpm",
        action='store_true',
        help="use DPM (2) sampler",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--H",
        type=int,
        default=512,
        help="image height, in pixel space",
    )
    parser.add_argument(
        "--W",
        type=int,
        default=512,
        help="image width, in pixel space",
    )
    parser.add_argument(
        "--C",
        type=int,
        default=4,
        help="latent channels",
    )
    parser.add_argument(
        "--f",
        type=int,
        default=8,
        help="downsampling factor, most often 8 or 16",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=12,
        help="how many prompts used in each batch"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=9.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file, separated by newlines",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/model_1.5.ckpt",
        help="path to the model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    parser.add_argument(
        "--device",
        type=str,
        help="Device on which Stable Diffusion will be run",
        choices=["cpu", "cuda"],
        default="cuda"
    )
    parser.add_argument(
        "--bf16",
        action='store_true',
        help="Use bfloat16",
    )
    # distributed generation
    parser.add_argument(
        "--n_gpus",
        type=int,
        default=1,
        help="number of gpus to use for generation",
    )
    parser.add_argument(
        "--gpu_idx",
        type=int,
        default=0,
        help="current gpu index",
    )
    parser.add_argument(
        "--n_nodes",
        type=int,
        default=1,
        help="number of nodes to use for generation",
    )
    parser.add_argument(
        "--node_idx",
        type=int,
        default=0,
        help="current node index",
    )
    opt = parser.parse_args()
    return opt


def main(opt):
    seed_everything(opt.seed)

    # data saver
    folder_name = ('plms' if opt.plms else 'ddim') + f'_{opt.scale}' + f'_seed_{opt.seed}'
    saver = ImageSaver(os.path.join(opt.outdir, folder_name), opt)

    # get the dataset and loader
    n_skip = opt.n_nodes * opt.n_gpus
    start = opt.node_idx * opt.n_gpus + opt.gpu_idx
    dataset = PromptDataset(file=opt.from_file, n_skip=n_skip, start=start, outdir=saver.outdir, opt=opt)
    data_loader = DataLoader(dataset,
                             batch_size=opt.batch_size,
                             shuffle=False,
                             num_workers=1)

    # get the model
    config = OmegaConf.load(f"{opt.config}")
    device = torch.device("cuda") if opt.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{opt.ckpt}", device)

    # get the generator
    generator = StableGenerator(model, opt)

    for (i, data) in enumerate(data_loader):
        prompts, ids = data[0], data[1]
        images = generator.generate(prompts, n_sample_per_prompt=opt.n_samples)

        # save images
        for j in range(len(images)):
            x_sample = images[j]
            img = Image.fromarray(x_sample.astype(np.uint8))
            if opt.img_save_size != opt.H:
                img = img.resize((opt.img_save_size, opt.img_save_size))
            saver.save(img, ids[j])


if __name__ == "__main__":
    opt = parse_args()
    main(opt)
