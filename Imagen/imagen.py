#!/usr/bin/env python3

import io
import math
import numbers
import os
import random
import re
import sys
import time
from collections import deque

import numpy as np
import torch
import webdataset as wds
import pickle

from functools import reduce
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as VTF
from torchvision.utils import make_grid, save_image
from PIL import Image
from tqdm import tqdm

from imagen_pytorch import ImagenTrainer, ElucidatedImagenConfig, ImagenConfig
from imagen_pytorch import load_imagen_from_checkpoint
from gan_utils import get_images, get_vocab
from data_generator import ImageLabelDataset

try:
    import wandb
except:
    pass


def safeget(dictionary, keys, default = None):
    return reduce(lambda d, key: d.get(key, default) if isinstance(d, dict) else default, keys.split('.'), dictionary)


def get_padding(image):    
    w, h = image.size
    max_wh = np.max([w, h])
    h_padding = (max_wh - w) / 2
    v_padding = (max_wh - h) / 2
    l_pad = h_padding if h_padding % 1 == 0 else h_padding+0.5
    t_pad = v_padding if v_padding % 1 == 0 else v_padding+0.5
    r_pad = h_padding if h_padding % 1 == 0 else h_padding-0.5
    b_pad = v_padding if v_padding % 1 == 0 else v_padding-0.5
    padding = (int(l_pad), int(t_pad), int(r_pad), int(b_pad))
    return padding


class PadImage(object):
    def __init__(self, fill=0, padding_mode='constant'):
        assert isinstance(fill, (numbers.Number, str, tuple))
        assert padding_mode in ['constant', 'edge', 'reflect', 'symmetric']

        self.fill = fill
        self.padding_mode = padding_mode

    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be padded.

        Returns:
            PIL Image: Padded image.
        """
        return VTF.pad(img, get_padding(img), self.fill, self.padding_mode)

    def __repr__(self):
        return self.__class__.__name__ + '(padding={0}, fill={1}, padding_mode={2})'.\
            format(self.fill, self.padding_mode)


def tuple_type(strings):
    strings = strings.replace("(", "").replace(")", "")
    mapped_int = map(int, strings.split(","))
    return tuple(mapped_int)


def main():
    import argparse
    import os
    # device_count = int(torch.cuda.device_count())
    # device_string = ""
    # for i in range(device_count):
    #     device_string += (f"{str(i)},")
    # device_string = device_string[:-1]
    # print(device_string)
    # os.environ["CUDA_LAUNCH_BLOCKING"] = '1'
    # os.environ["CUDA_VISIBLE_DEVICES"] = device_string

    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default=None, help="image source")
    parser.add_argument('--tags_source', type=str, default=None, help="tag files. will use --source if not specified.")
    parser.add_argument('--cond_images', type=str, default=None)
    parser.add_argument('--style', type=str, default=None)
    parser.add_argument('--embeddings', type=str, default=None)
    parser.add_argument('--tags', type=str, default=None)
    parser.add_argument('--vocab', default=None)
    parser.add_argument('--size', default=256, type=int)
    parser.add_argument('--sample_steps', default=256, type=int)
    parser.add_argument('--num_unets', default=1, type=int, help="additional unet networks")
    parser.add_argument('--vocab_limit', default=None, type=int)
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--imagen', default="imagen.pth")
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--replace', action='store_true', help="replace the output file")
    parser.add_argument('--unet_dims', default=256, type=int)
    parser.add_argument('--unet2_dims', default=256, type=int)
    parser.add_argument('--dim_mults', default="(1,2,3,4)", type=tuple_type)
    parser.add_argument("--start_size", default=64, type=int)
    parser.add_argument("--sample_unet", default=1, type=int)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--text_encoder', type=str, default="t5-large")
    parser.add_argument("--cond_scale", default=7.5, type=float, help="sampling conditional scale 0-10.0")
    parser.add_argument('--no_elu', action='store_true', help="don't use elucidated imagen")
    parser.add_argument("--num_samples", default=1, type=int, help="")
    parser.add_argument("--init_image", default=None,)
    parser.add_argument("--skip_steps", default=None, type=int)
    parser.add_argument("--sigma_max", default=80, type=float)
    parser.add_argument("--full_load", action="store_true",
                        help="don't use load_from_checkpoint.")
    parser.add_argument('--no_memory_efficient', action='store_true',
                        help="don't use memory_efficient unet1")
    parser.add_argument('--print_params', action='store_true',
                        help="print model params and exit")
    parser.add_argument("--unet_size_mult", default=4, type=int)
    parser.add_argument("--self_cond", action="store_true")

    # training
    parser.add_argument('--batch_size', default=4, type=int)
    parser.add_argument('--micro_batch_size', default=4, type=int)
    parser.add_argument('--samples_out', default="samples")
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--train_encoder', action='store_true')
    parser.add_argument('--shuffle_tags', action='store_true')
    parser.add_argument('--train_unet', type=int, default=2)
    parser.add_argument('--random_drop_tags', type=float, default=0.3)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--no_text_transform', action='store_true')
    parser.add_argument('--start_epoch', default=1, type=int)
    parser.add_argument('--no_patching', action='store_true')
    parser.add_argument('--create_embeddings', action='store_true')
    parser.add_argument('--verify_images', action='store_true')
    parser.add_argument('--pretrained', default="t5-large")
    parser.add_argument('--no_sample', action='store_true',
                        help="do not sample while training")
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument('--loss', default="l2")
    parser.add_argument('--sample_rate', default=2500, type=int)
    parser.add_argument('--wandb', action='store_true',
                        help="use wandb logging")
    parser.add_argument('--is_t5', action='store_true',
                        help="t5-like encoder")
    parser.add_argument('--webdataset', action='store_true')

    # Sicong: for inpainting test
    parser.add_argument('--test_pkl', type=str, default=None, help="pickle file needed to do inpainting during inference")

    args = parser.parse_args()

    if args.sample_steps is None:
        args.sample_steps = args.size

    if args.tags_source is None:
        args.tags_source = args.source

    if args.vocab is None:
        args.vocab = args.source
    else:
        assert os.path.isfile(args.vocab) or os.path.isdir(args.vocab)

    if args.bf16:
        # probably (maybe) need to set TORCH_CUDNN_V8_API_ENABLED=1 in environment
        if args.device == "cuda":
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("medium")

    if args.print_params:
        print_model_params(args)
        sys.exit()

    if args.wandb:
        wandb.init(project=os.path.splitext(os.path.basename(args.imagen))[0])

    if args.create_embeddings:
        create_embeddings(args)

    if args.train_encoder:
        train_encoder(args)

    if args.train:
        train(args)
    else:
        sample(args)


def sample(args):

    if os.path.isfile(args.output) and not args.replace:
        return

    try:
        imagen = load(args).to(args.device)
    except Exception as ex:
        print(f"Error loading model: {args.imagen}")
        print(ex)
        return

    args.num_unets = len(imagen.unets) - 1

    image_sizes = get_image_sizes(args)
    print(f"image sizes: {image_sizes}")

    imagen.image_sizes = image_sizes

    cond_image = None

    if args.cond_images is not None and os.path.isfile(args.cond_images):
        tforms = transforms.Compose([PadImage(),
                                     transforms.Resize((args.size, args.size)),
                                     transforms.ToTensor()])
        cond_image = Image.open(args.cond_images)
        cond_image = tforms(cond_image).to(imagen.device)
        cond_image = torch.unsqueeze(cond_image, 0)
        cond_image = cond_image.repeat(args.num_samples, 1, 1, 1).to(args.device)

    init_image = None

    if args.init_image is not None:
        tforms = transforms.Compose([PadImage(),
                                     transforms.Resize((args.size, args.size)),
                                     transforms.ToTensor()])
        init_image = Image.open(args.init_image)
        init_image = tforms(init_image).to(imagen.device)
        init_image = torch.unsqueeze(init_image, 0)
        init_image = init_image.repeat(args.num_samples, 1, 1, 1).to(args.device)

    style_image = None

    if args.style is not None and os.path.isfile(args.style):
        tforms = transforms.Compose([PadImage(),
                                     transforms.Resize((args.size, args.size)),
                                     transforms.ToTensor()])
        style_image = Image.open(args.style)
        style_image = tforms(style_image).to(imagen.device)
        style_image = torch.unsqueeze(style_image, 0)
        style_image = style_image.repeat(args.num_samples, 1, 1, 1).to(args.device)

    text_embeds = None
    # sample_texts = args.tags
    # inpaint_text = 'The master room should be at north side with about 200 sqft and the aspect ratio of 8 over 9. The master room should have an en-suite bathroom. Can you make bathroom at south west corner with around 50 sqft and the aspect ratio of 9 over 7? The bathroom can be used by guest. Can you make living room  with approx 800 sqft and the aspect ratio of 5 over 12? Make common room at south side with around 150 sqft and the aspect ratio of 4 over 3. The common room should have an en-suite bathroom. I would like to have kitchen at south side with approx 50 sqft and the aspect ratio of 15 over 16. Make balcony at north side with approx 100 sqft and the aspect ratio of 13 over 5.'
    # sample_texts = [inpaint_text]*2

    # Sicong: load pkl file for inpainting
    with (open(args.test_pkl, "rb")) as openfile:
        sample_texts = pickle.load(openfile)
    print(f"{len(sample_texts)} sentences are loaded from {args.test_pkl} for generating sample images.")

    if args.embeddings is not None:
        sample_texts = args.embeddings
        text_embeds = get_text_embeddings(sample_texts, text_encoder=args.text_encoder)
        text_embeds = torch.from_numpy(text_embeds)
        text_embeds = torch.tile(text_embeds, (args.num_samples, 1))
        text_embeds = torch.unsqueeze(text_embeds, 1).to(args.device)
        sample_texts = None
    else:
        # sample_texts = list(np.repeat(sample_texts, args.num_samples))
        pass
    
    img_ids = list(sample_texts.keys())
    for index in range(len(img_ids)//12):
        if 12*(index+1) > len(img_ids):
            batch_img_ids = img_ids[12*index:]
        else:
            batch_img_ids = img_ids[12*index:12*(index+1)]
        print(f"sampling image with id {batch_img_ids} ... process: {12*index}/{len(sample_texts)}")
        sample_text_ = [sample_texts[b_id] for b_id in batch_img_ids]
        sample_images = imagen.sample(texts=sample_text_,
                                    text_embeds=text_embeds,
                                    cond_images=cond_image,
                                    cond_scale=args.cond_scale,
                                    init_images=init_image,
                                    skip_steps=args.skip_steps,
                                    sigma_max=args.sigma_max,
                                    return_pil_images=True,
                                    stop_at_unet_number=args.sample_unet)
        for i, sample in enumerate(sample_images):
            final_image = sample
            bn, ext = os.path.splitext(args.output)
            output_file = bn + f"_{batch_img_ids[i]}" + ext
            final_image.resize((args.size, args.size)).save(output_file)

    # for id, img_id in enumerate(sample_texts):
    #     print(f"sampling image with id{img_id}... process: {id}/{len(sample_texts)}")
    #     sample_text_ = [sample_texts[img_id]]
    #     # sample_texts = list(np.repeat(sample_texts, args.num_samples))
    #     sample_images = imagen.sample(texts=sample_text_,
    #                                 text_embeds=text_embeds,
    #                                 cond_images=cond_image,
    #                                 cond_scale=args.cond_scale,
    #                                 init_images=init_image,
    #                                 skip_steps=args.skip_steps,
    #                                 sigma_max=args.sigma_max,
    #                                 return_pil_images=True,
    #                                 stop_at_unet_number=args.sample_unet)

    #     for i, sample in enumerate(sample_images):
    #         final_image = sample
    #         bn, ext = os.path.splitext(args.output)
    #         output_file = bn + f"_{img_id}_{i}" + ext

    #         if args.num_samples == 1:
    #             output_file = bn + f"_{img_id}" + ext

    #         final_image.resize((args.size, args.size)).save(output_file)


def restore_parts(state_dict_target, state_dict_from):
    for name, param in state_dict_from.items():
        if name not in state_dict_target:
            continue
        # if isinstance(param, Parameter):
        #    param = param.data
        if param.size() == state_dict_target[name].size():
            state_dict_target[name].copy_(param)
        else:
            print(f"layer {name}({param.size()} different than target: {state_dict_target[name].size()}")

    return state_dict_target


def save(imagen, path):
    out = {}
    unets = []
    for unet in imagen.unets:
        unets.append(unet.cpu().state_dict())
    out["unets"] = unets

    out["imagen"] = imagen.cpu().state_dict()

    torch.save(out, path)


def print_model_params(args):
    loaded = torch.load(args.imagen, map_location="cpu")
    imagen_params = safeget(loaded, 'imagen_params')

    print(imagen_params)


def load(args):

    if not args.full_load:
        imagen = load_imagen_from_checkpoint(args.imagen)

    else:
        model = torch.load(args.imagen, map_location="cpu")["model"]

        imagen = get_imagen(args)
        imagen.load_state_dict(model)

    return imagen


def get_image_sizes(args):
    image_sizes = [args.start_size]

    for i in range(0, args.num_unets):
        ns = image_sizes[-1] * args.unet_size_mult
        if args.train and not args.no_patching:
            ns = ns // args.unet_size_mult
        image_sizes.append(ns)

    image_sizes[-1] = args.size // args.unet_size_mult if args.train and not args.no_patching else args.size

    return image_sizes


def get_imagen(args, unet_dims=None, unet2_dims=None):

    if unet_dims is None:
        unet_dims = args.unet_dims

    if unet2_dims is None:
        unet2_dims = args.unet2_dims

    if args.cond_images is not None:
        cond_images_channels = 3
    else:
        cond_images_channels = 0

    # unet for imagen
    unet1 = dict(
        dim=unet_dims,
        cond_dim=512,
        dim_mults=args.dim_mults,
        cond_images_channels=cond_images_channels,
        num_resnet_blocks=2,
        layer_attns=(False, True, True, True),
        layer_cross_attns=(False, True, True, True),
        use_global_context_attn=False,
        attn_pool_text=False,
        memory_efficient=not args.no_memory_efficient,
        self_cond=args.self_cond
    )

    unets = [unet1]

    for i in range(args.num_unets):

        unet2 = dict(
            dim=unet2_dims // (i + 1),
            cond_dim=512,
            dim_mults=(1, 2, 3, 4),
            cond_images_channels=cond_images_channels,
            num_resnet_blocks=2,
            layer_attns=(False, False, False, i < 2),
            layer_cross_attns=(False, False, True, True),
            use_global_context_attn=False,
            # final_conv_kernel_size=1,
            attn_pool_text=False,
            memory_efficient=True,
            self_cond=args.self_cond
        )

        unets.append(unet2)

    image_sizes = get_image_sizes(args)

    print(f"image_sizes={image_sizes}")

    sample_steps = args.sample_steps # [args.sample_steps] * (args.num_unets + 1)

    if not args.no_elu:
        imagen = ElucidatedImagenConfig(
            unets=unets,
            text_encoder_name=args.text_encoder,
            num_sample_steps=sample_steps,
            lowres_noise_schedule="cosine",
            # pred_objectives=["noise", "x_start"],
            image_sizes=image_sizes,
            per_sample_random_aug_noise_level=True,
            sigma_max = args.sigma_max,
            cond_drop_prob=0
        ).create().to(args.device)

    else:
        imagen = ImagenConfig(
            unets=unets,
            text_encoder_name=args.text_encoder,
            noise_schedules=["cosine", "cosine"],
            pred_objectives=["noise", "x_start"],
            image_sizes=image_sizes,
            per_sample_random_aug_noise_level=True,
            lowres_sample_noise_level=0.3,
            loss_type=args.loss
        ).create().to(args.device)

    return imagen


def make_training_samples(cond_images, styles, trainer, args, epoch, step, epoch_loss):
    
    # sample_texts = ['1girl, red_bikini, bikini, swimsuit, outdoors, pool, brown_hair',
    #                 '1girl, blue_dress, eyes_closed, blonde_hair',
    #                 '1boy, black_hair',
    #                 '1girl, wristwatch, red_hair']
    # with (open(args.test_pkl, "rb")) as openfile:
    #     data = pickle.load(openfile)
    #     inpaint_text = data['text']
    #     inpaint_mask = data['inpaint_mask'].unsqueeze(0).cuda()
    #     inpaint_init_image = data['inpaint_image']
    # # inpaint_images = torch.randn(1, 3, 256, 256).cuda()
    # inpaint_images = torch.empty(4, 3, 256, 256).fill_(255.).cuda()
    # inpaint_masks = torch.cat([inpaint_mask]*4, dim=0)

    inpaint_text = 'The master room should be at north side with about 200 sqft and the aspect ratio of 8 over 9. The master room should have an en-suite bathroom. Can you make bathroom at south west corner with around 50 sqft and the aspect ratio of 9 over 7? The bathroom can be used by guest. Can you make living room  with approx 800 sqft and the aspect ratio of 5 over 12? Make common room at south side with around 150 sqft and the aspect ratio of 4 over 3. The common room should have an en-suite bathroom. I would like to have kitchen at south side with approx 50 sqft and the aspect ratio of 15 over 16. Make balcony at north side with approx 100 sqft and the aspect ratio of 13 over 5.'
    sample_texts = [inpaint_text]*4 

    # trainer.accelerator.wait_for_everyone()

    if args.device == "cuda":
        torch.cuda.empty_cache()

    disp_size = min(args.batch_size, 4)
    
    sample_cond_images = None
    sample_style_images = None

    if cond_images is not None:
        sample_cond_images = cond_images[:disp_size]

    if styles is not None:
        sample_style_images = styles[:disp_size]

    # dup the sampler's image sizes temporarily:
    args.train = False
    sample_image_sizes = get_image_sizes(args)
    args.train = True

    train_image_sizes = trainer.imagen.image_sizes

    trainer.imagen.image_sizes = sample_image_sizes

    text_embeds = None
    if args.embeddings is not None:
        text_embeds = get_text_embeddings(sample_texts, text_encoder=args.text_encoder)
        text_embeds = torch.from_numpy(text_embeds)
        text_embeds = torch.unsqueeze(text_embeds, 1)
        sample_texts = None

    # if trainer.accelerator.is_main_process:
    #     print("1")
    #     with torch.no_grad():
    #         imagen_eval = get_imagen(args)
    #         if args.imagen is not None and os.path.isfile(args.imagen):
    #             print("0")
    #             print(f"Loading model: {args.imagen}")
    #             with trainer.fs.open(args.imagen) as f:
    #                 loaded_obj = torch.load(f, map_location='cpu')
    #             imagen_eval.load_state_dict(loaded_obj['model'])
    #             sample_images = imagen_eval.sample(texts=sample_texts,
    #                                     cond_scale=args.cond_scale,
    #                                     stop_at_unet_number=args.train_unet)
    #             # sample_images = trainer.accelerator.gather(sample_images)
    # # restore train image sizes:
    # trainer.imagen.image_sizes = train_image_sizes

    
    sample_images = trainer.sample(texts=sample_texts,
                                    text_embeds=text_embeds,
                                    cond_images=sample_cond_images,
                                    cond_scale=args.cond_scale,
                                    return_all_unet_outputs=True,
                                    stop_at_unet_number=args.train_unet)
    # restore train image sizes:
    trainer.imagen.image_sizes = train_image_sizes
    
    # sample_images = trainer.sample(texts=sample_texts,
    #                                    text_embeds=text_embeds,
    #                                    cond_images=sample_cond_images,
    #                                    cond_scale=args.cond_scale,
    #                                    return_all_unet_outputs=True,
    #                                    stop_at_unet_number=args.train_unet)

    # 

    final_samples = None

    if len(sample_images) > 1:
        for si in sample_images:
            sample_images1 = transforms.Resize(args.size)(si)
            if final_samples is None:
                final_samples = sample_images1
                continue

            sample_images1 = transforms.Resize(args.size)(si)
            final_samples = torch.cat([final_samples, sample_images1])

        sample_images = final_samples
    else:
        sample_images = sample_images[0]
        sample_images = transforms.Resize(args.size)(sample_images)

    if cond_images is not None:
        sample_poses0 = transforms.Resize(args.size)(sample_cond_images)
        sample_images = torch.cat([sample_images.cpu(), sample_poses0.cpu()])

    if styles is not None:
        sample_poses0 = transforms.Resize(args.size)(sample_style_images)
        sample_images = torch.cat([sample_images.cpu(), sample_poses0.cpu()])

    print(sample_images.size())
    grid = make_grid(sample_images, nrow=disp_size, normalize=False, range=(-1, 1))
    VTF.to_pil_image(grid).save(os.path.join(args.samples_out, f"imagen_{epoch}_{int(step / epoch)}_loss{epoch_loss}.png"))

    # sicong: log image to wandb
    wandb.log({"example": wandb.Image(os.path.join(args.samples_out, f"imagen_{epoch}_{int(step / epoch)}_loss{epoch_loss}.png"),caption=f"imagen_{epoch}_{int(step / epoch)}_loss{epoch_loss}")})


def delete_random_elems(input_list, n):
    to_delete = set(random.sample(range(len(input_list)), n))
    return [x for i,x in enumerate(input_list) if not i in to_delete]

def my_split_by_node(urls):
    node_id, node_count = torch.distributed.get_rank(), torch.distributed.get_world_size()
    return urls[node_id::node_count]

def create_webdataset(
                urls,
                image_transform,
                txt_transform,
                enable_text=True,
                enable_image=True,
                image_key='jpg',
                caption_key='txt',
                cache_path=None,):

    dataset = wds.WebDataset(urls,
                             nodesplitter=wds.split_by_node,
                             cache_dir=cache_path,
                             cache_size=10**10,
                             handler=wds.handlers.warn_and_continue)

    def preprocess_dataset(item):
        # print(item.keys())
        if enable_image:
            image_data = item[image_key]
            image = Image.open(io.BytesIO(image_data))
            image_tensor = image_transform(image)
            
        if enable_text:
            text = item[caption_key]
            caption = text.decode("utf-8") 
            transformed_text = txt_transform(caption)

        return (image_tensor, transformed_text)

    transformed_dataset = dataset.shuffle(1000).map(preprocess_dataset, handler=wds.handlers.warn_and_continue)
    return transformed_dataset

def train(args):

    imagen = get_imagen(args)

    precision = None

    if args.fp16:
        precision = "fp16"
    elif args.bf16:
        precision = "bf16"

    trainer = ImagenTrainer(imagen, precision=precision, lr=args.lr)
    # print(trainer.fs)

    if args.imagen is not None and os.path.isfile(args.imagen):
        print(f"Loading model: {args.imagen}")
        trainer.load(args.imagen, only_model=args.full_load) 

    print(f"Fetching image indexes in {args.source}...")

    if not args.webdataset:
        imgs = get_images(args.source, verify=args.verify_images)

        if args.embeddings is not None:
            txts = get_images(args.embeddings, exts=".npz")
        else:
            txts = get_images(args.tags_source, exts=".txt")

        print(f"{len(imgs)} images")
        print(f"{len(txts)} tags")

    cond_images = None
    has_cond = False
    style_images = None
    has_style = False

    if args.cond_images is not None:
        cond_images = get_images(args.cond_images)
        print(f"{len(cond_images)} conditional images")
        has_cond = True

    if args.style is not None:
        style_images = get_images(args.style)
        print(f"{len(style_images)} style images")
        has_style = True

    # get non-training sizes for image resizing/cropping
    args.train = False
    train_img_size = get_image_sizes(args)[args.train_unet - 1]

    tforms = transforms.Compose([
            PadImage(),
            transforms.Resize(train_img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()])

    alt_tforms = transforms.Compose([
            PadImage(),
            transforms.Resize(train_img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor()])

    if args.train_unet > 1 and not args.no_patching:
        tforms = transforms.Compose([
            transforms.Resize(args.size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.RandomCrop(train_img_size),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor()])

    def txt_xforms(txt):
        # print(f"txt: {txt}")
        # txt = txt.replace("_", " ")
        txt = txt.split(", ")
        if args.shuffle_tags:
            np.random.shuffle(txt)

        r = int(len(txt) * args.random_drop_tags)

        if r > 0:
            r = random.randrange(r)

        if args.random_drop_tags > 0.0 and r > 0:
            txt = delete_random_elems(txt, r)

        txt = ", ".join(txt)

        return txt
    
    ## Sicong: dont transform the text
    # tag_transform = txt_xforms
    tag_transform = None

    if args.no_text_transform:
        tag_transform = None

    if args.webdataset:
        data = create_webdataset(args.source, tforms, txt_xforms)
        dl = torch.utils.data.DataLoader(data,
                                         batch_size=args.batch_size,
                                         num_workers=args.workers)
    else:
        data = ImageLabelDataset(imgs, txts, None,
                                 styles=style_images,
                                 cond_images=cond_images,
                                 dim=(args.size, args.size),
                                 transform=tforms,
                                 alt_transform=alt_tforms,
                                 tag_transform=tag_transform,
                                 channels_first=True,
                                 return_raw_txt=True,
                                 no_preload=True,
                                 use_text_encodings=args.embeddings is not None)

        dl = torch.utils.data.DataLoader(data,
                                         batch_size=args.batch_size,
                                         shuffle=True,
                                         num_workers=args.workers,
                                         pin_memory=True)

    dl = trainer.accelerator.prepare(dl)

    os.makedirs(args.samples_out, exist_ok=True)

    for epoch in range(args.start_epoch, args.epochs + 1):
        step = 0
        epoch_loss = 0
        with tqdm(dl, unit="batches", disable=not trainer.accelerator.is_local_main_process) as tepoch:
            for data in tepoch:

                cond_images = None
                style_images = None

                images = data.pop(0)
                texts = data.pop(0)

                if has_style:
                    style_images = data.pop(0)

                if has_cond:
                    cond_images = data.pop(0)

                step += 1

                txt_embeds = None

                if args.embeddings is not None:
                    txt_embeds = texts
                    texts = None

                # print(txt_embeds.size())
                try:
                    loss = trainer(
                        images,
                        cond_images=cond_images,
                        texts=texts,
                        text_embeds=txt_embeds,
                        unet_number=args.train_unet,
                        max_batch_size=args.micro_batch_size
                    )
                except ValueError as ve:
                    print(ve)
                    print(texts)

                trainer.update(unet_number=args.train_unet)

                epoch_loss += loss
                epoch_loss_disp = round(float(epoch_loss) / float(step), 6)

                tepoch.set_description(f"Epoch {epoch}")
                tepoch.set_postfix(loss=round(loss, 6), epoch_loss=epoch_loss_disp)

                if args.wandb:
                    wandb.log({"loss": loss, "epoch_loss": epoch_loss_disp})
                
                # if step % args.sample_rate == 0:
                #     if trainer.accelerator.is_main_process:
                #         if not args.no_sample:
                #             make_training_samples(cond_images, style_images, trainer, args, epoch,
                #                                 trainer.num_steps_taken(args.train_unet),
                #                                 epoch_loss_disp)
                #     # if args.imagen is not None:
                #     #     trainer.save(args.imagen)
                
                # trainer.accelerator.wait_for_everyone()
        # END OF EPOCH
        if args.imagen is not None:
            trainer.save(args.imagen)

        if not args.no_sample:
            if trainer.accelerator.is_main_process:
                make_training_samples(cond_images, style_images, trainer, args, epoch,
                                  trainer.num_steps_taken(args.train_unet),
                                  epoch_loss_disp)

        
        trainer.accelerator.wait_for_everyone()

        # if epoch % 8 == 0:
        #     path = args.imagen.split('.pth')[0] + "_ep" + str(epoch) + ".pth"
        #     trainer.save(f"{path}")
        #     # wandb.save(path)

        #     if args.device == "cuda":
        #         # prevents OOM on memory constrained devices
        #         torch.cuda.empty_cache()


class LineByLineTextDataset(Dataset):
    def __init__(self, tokenizer, file_paths: str, block_size=512):

        lines = []

        for file_path in tqdm(file_paths):
            assert os.path.isfile(file_path)
            with open(file_path, encoding="utf-8") as f:
                lines.extend([line for line in f.read().splitlines() if (len(line) > 0 and not line.isspace())])

        if tokenizer is not None:
            self.examples = tokenizer.batch_encode_plus(lines, add_special_tokens=True, max_length=block_size)["input_ids"]
        else:
            self.examples = lines

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


def train_tokenizer(args):
    import io
    import sentencepiece as spm

    if args.vocab is None:
        args.vocab = args.tags_source

    output_file = os.path.join(args.text_encoder, "t5_model.spm")
    if os.path.isfile(output_file):
        return output_file

    os.makedirs(args.text_encoder, exist_ok=True)

    print("Fetching vocab...")

    vocab = get_vocab(args.vocab, top=args.vocab_limit)

    # save vocab
    if not os.path.isfile(args.vocab):
        with open(os.path.join(args.text_encoder, "vocab.txt"), 'w') as f:
            f.write(", ".join(vocab))

    vocab_file = os.path.join(args.text_encoder, "t5_vocab_input.tsv")

    with open(vocab_file, 'w') as f:
        f.write("\n".join(vocab))

        # for i, w in enumerate(vocab):
        #     f.write(f"{w}\t{i}\n")

    print(f"vocab size: {len(vocab)}")
    print("training tokenizer...")
    model = io.BytesIO()
    spm.SentencePieceTrainer.train(input=vocab_file,
                                   input_format="text",
                                   model_writer=model,
                                   input_sentence_size=6000000,
                                   max_sentence_length=16384,
                                   max_sentencepiece_length=96,
                                   shuffle_input_sentence=True,
                                   split_by_unicode_script=False,
                                   split_by_whitespace=True,
                                   split_digits=False,
                                   num_threads=8,
                                   pad_id=0,
                                   eos_id=1,
                                   unk_id=2,
                                   bos_id=3,
                                   model_type='unigram',
                                   vocab_size=len(vocab) // 2)

    with open(output_file, 'wb') as f:
        f.write(model.getvalue())

    return output_file


def train_encoder(args):

    from transformers import T5ForConditionalGeneration, TrainingArguments, Trainer
    from transformers import DataCollatorForLanguageModeling
    from transformers import T5Tokenizer

    assert args.text_encoder is not None

    pretrained = args.pretrained

    model = T5ForConditionalGeneration.from_pretrained(pretrained)

    t5_spm_model = train_tokenizer(args)

    tokenizer = T5Tokenizer(t5_spm_model)

    txts = get_images(args.tags_source, exts=".txt")
    tokenizer.save_pretrained(args.text_encoder)

    tokenizer.pad_token = tokenizer.eos_token

    lm_dataset = LineByLineTextDataset(tokenizer, txts)
    val_dataset = LineByLineTextDataset(tokenizer, txts[-2:])

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    resume_from_checkpoint = None

    if os.path.isfile(os.path.join(args.text_encoder, "config.json")):
        resume_from_checkpoint = args.text_encoder

    training_args = TrainingArguments(
                                      output_dir=args.text_encoder,
                                      evaluation_strategy="epoch",
                                      learning_rate=2e-5,
                                      weight_decay=0.01,
                                      num_train_epochs=args.epochs,
                                      auto_find_batch_size=True,
                                      save_strategy="epoch",
                                      save_total_limit=3,
                                      bf16=args.bf16
                                     )

    trainer = Trainer(
                      model=model,
                      args=training_args,
                      train_dataset=lm_dataset,
                      eval_dataset=val_dataset,
                      data_collator=data_collator,
                     )
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except KeyboardInterrupt:
        print("Training cancelled by user.")

    print("saving model...")
    trainer.save_model(args.text_encoder)


def get_text_embeddings(txts, tokenizer=None, model=None, text_encoder=None):
    from transformers import AutoModel, AutoTokenizer

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(text_encoder)
        model = AutoModel.from_pretrained(text_encoder)
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token

    toks = tokenizer(txts, padding=True, truncation=True, return_tensors="pt").to(model.device)

    with torch.no_grad():
        last_hidden_state = model(**toks, output_hidden_states=True, return_dict=True).last_hidden_state


    weights = torch.arange(start=1, end=last_hidden_state.shape[1] + 1).unsqueeze(-1).expand(last_hidden_state.size()).float()
    weights = weights.to(last_hidden_state.device)

    input_mask_expanded = toks["attention_mask"].unsqueeze(-1).expand(last_hidden_state.size()).float()

    sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
    sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

    embeddings = sum_embeddings / sum_mask
    embeddings = embeddings.cpu().detach().numpy()

    return embeddings


def get_text_embeddings_t5(txts, text_encoder):
    from imagen_pytorch.t5 import t5_encode_text

    return t5_encode_text(txts, name=text_encoder).cpu().detach().numpy()


def create_embeddings(args):
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    model = AutoModel.from_pretrained(args.text_encoder).to(args.device)

    print("fetching tags...")

    txts = get_images(args.tags_source, exts=".txt")

    tokenizer.padding_side = "right"
    tokenizer.pad_token = tokenizer.eos_token

    empties = []

    if args.output is not None:
        os.makedirs(args.output, exist_ok=True)

    print("encoding...")

    for txt in tqdm(txts):

        basepath = os.path.dirname(txt)
        bn = os.path.splitext(os.path.basename(txt))[0]

        if args.output is None:
            out_file = os.path.join(basepath, f"{bn}.npz")
        else:
            out_file = os.path.join(args.output, f"{bn}.npz")

        if os.path.isfile(out_file) and not args.replace:
            continue

        with open(txt, 'r') as f:
            data = f.read()

        if data == "":
            empties.append(txt)
            continue

        if args.is_t5:
            embeddings = get_text_embeddings_t5(data, args.text_encoder)
            print(embeddings.shape)
        else:
            embeddings = get_text_embeddings(data, tokenizer, model)

        np.savez_compressed(out_file, embeddings)

    with open("empties.txt", 'w') as f:
        f.write("\n".join(empties))

if __name__ == "__main__":
    main()
