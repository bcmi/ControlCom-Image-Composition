from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from audioop import reverse
from cmath import inf
from curses.panel import bottom_panel
from dis import dis
from email.mime import image
import os
from io import BytesIO
import logging
import base64
from sre_parse import State
from sys import prefix
import threading
import random
from turtle import left, right
import numpy as np
from typing import Callable, List, Tuple, Union
from PIL import Image,ImageDraw,ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
# from sympy import source
import torch.utils.data as data
import json
import time
import cv2
cv2.setNumThreads(0)
import torch
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
import copy
import math
from functools import partial
from tqdm import tqdm
import sys
import shutil
import transformers
import gc

proj_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, proj_dir)
# import objgraph
# from memory_profiler import profile
# import tracemalloc
# torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 300
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1" 

def bbox_process(bbox):
    x_min = int(bbox[0])
    y_min = int(bbox[1])
    x_max = x_min + int(bbox[2])
    y_max = y_min + int(bbox[3])
    return list(map(int, [x_min, y_min, x_max, y_max]))

def get_tensor(normalize=True, toTensor=True, resize=True, image_size=(512, 512)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.5, 0.5, 0.5),
                                                (0.5, 0.5, 0.5))]
    return torchvision.transforms.Compose(transform_list)

def get_tensor_clip(normalize=True, toTensor=True, resize=True, image_size=(224, 224)):
    transform_list = []
    if resize:
        transform_list += [torchvision.transforms.Resize(image_size)]
    if toTensor:
        transform_list += [torchvision.transforms.ToTensor()]
    if normalize:
        transform_list += [torchvision.transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
        (0.26862954, 0.26130258, 0.27577711))]
    return torchvision.transforms.Compose(transform_list)

def scan_all_files():
    bbox_dir = os.path.join(proj_dir, '../../dataset/open-images/bbox_mask')
    assert os.path.exists(bbox_dir), bbox_dir
    
    bad_files = []
    for split in os.listdir(bbox_dir):
        total_images, total_pairs, bad_masks, bad_images = 0, 0, 0, 0
        subdir = os.path.join(bbox_dir, split)
        if not os.path.isdir(subdir) or split not in ['train', 'test', 'validation']:
            continue
        for file in tqdm(os.listdir(subdir)):
            try:
                with open(os.path.join(subdir, file), 'r') as f:
                    for line in f.readlines():
                        line = line.strip()
                        info = line.split(' ')
                        mask_file = os.path.join(bbox_dir, '../masks', split, info[-2])
                        if os.path.exists(mask_file):
                            total_pairs += 1
                        else:
                            bad_masks += 1
                total_images += 1
            except:
                bad_files.append(file)
                bad_images += 1
        print('{}, {} images({} bad), {} pairs({} bad)'.format(
            split, total_images, bad_images, total_pairs, bad_masks))
        
        if len(bad_files) > 0:
            with open(os.path.join(bbox_dir, 'bad_files.txt'), 'w') as f:
                for file in bad_files:
                    f.write(file + '\n')
        
    print(f'{len(bad_files)} bad_files')

class DataAugmentation:
    def __init__(self, augment_background, augment_bbox, border_mode=0):
        self.blur = A.Blur(p=0.3)
        self.appearance_trans = A.Compose([
            A.ColorJitter(brightness=0.5, 
                          contrast=0.5, 
                          saturation=0.5, 
                          hue=0.05, 
                          always_apply=False, 
                          p=1)],
            additional_targets={'image':'image', 'image1':'image', 'image2':'image'}
            )
        self.geometric_trans = A.Compose([
            A.HorizontalFlip(p=0.15),
            A.Rotate(limit=30,
                     border_mode=border_mode,
                     value=(127,127,127),
                     mask_value=0,
                     p=1),
            A.Perspective(scale=(0.05, 0.25), 
                          pad_mode=border_mode,
                          pad_val =(127,127,127),
                          mask_pad_val=0,
                          fit_output=False, 
                          p=1)
        ])
        self.crop_bg_p  = 0.5
        self.pad_bbox_p = 0.5 if augment_bbox else 0
        self.augment_background_p = 0.3 if augment_background else 0
        self.bbox_maxlen = 0.7
    
    def __call__(self, bg_img, inpaint_img, bbox, bg_mask, fg_img, fg_mask, indicator, new_bg):
        # randomly crop background image
        if self.crop_bg_p > 0 and np.random.rand() < self.crop_bg_p:
            crop_bg, crop_inpaint, crop_bbox, crop_mask = self.random_crop_background(bg_img, inpaint_img, bbox, bg_mask)
        else:
            crop_bg, crop_inpaint, crop_bbox, crop_mask = bg_img, inpaint_img, bbox, bg_mask
        # randomly pad bounding box of foreground
        if self.pad_bbox_p > 0 and np.random.rand() < self.pad_bbox_p:
            pad_bbox = self.random_pad_bbox(crop_bbox, crop_bg.shape[1], crop_bg.shape[0])
        else:
            pad_bbox = crop_bbox
        pad_mask = bbox2mask(pad_bbox, crop_bg.shape[1], crop_bg.shape[0])
        # perform illumination transformation on background
        if indicator[0] == 0 and self.augment_background_p > 0 and np.random.rand() < self.augment_background_p:
            trans_imgs = self.appearance_trans(image=crop_bg.copy(), 
                                               image1=crop_inpaint.copy())
            trans_bg = trans_imgs['image']
            trans_inpaint = trans_imgs['image1']
        else:
            trans_bg = crop_bg.copy()
            trans_inpaint = crop_inpaint.copy()
        # perform illumination and pose transformation on foreground
        app_trans_fg, geo_trans_fg, trans_fgmask = self.augment_foreground(fg_img.copy(), fg_mask.copy(), indicator, new_bg)
        trans_fg = app_trans_fg if indicator[1] == 0 else geo_trans_fg
        # generate composite by copy-and-paste foreground object
        if indicator[0] == 0:
            x1,y1,x2,y2 = crop_bbox
            trans_bg[y1:y2,x1:x2] = np.where(fg_mask[:,:,np.newaxis] > 127, app_trans_fg, trans_bg[y1:y2,x1:x2])
        transformed = self.blur(image=trans_fg)
        trans_fg = transformed['image']
        transformed = None
        return {"bg_img":   trans_bg,
                "bg_mask":  crop_mask,
                "inpaint_img": trans_inpaint,
                "bbox":     crop_bbox,
                "pad_bbox": pad_bbox,
                "pad_mask": pad_mask,
                "fg_img":   trans_fg,
                "fg_mask":  trans_fgmask,
                "gt_fg_mask": fg_mask}
    
    # @func_set_timeout(0.1)
    def perform_geometry_augmentation(self, app_img, trans_mask, new_bg):
        # geometric transformed image
        transformed = self.geometric_trans(image=app_img, mask=trans_mask)
        geo_img    = transformed['image']
        trans_mask = transformed['mask']
        if new_bg is not None:
            geo_img = np.where(trans_mask[:,:,np.newaxis] > 127, geo_img, new_bg)
        return geo_img, trans_mask
    
    def augment_foreground(self, img, mask, indicator, new_bg):
        # appearance transformed image
        if new_bg is None:
            transformed = self.appearance_trans(image=img)
        else:
            transformed = self.appearance_trans(image=img, image1=new_bg)
            new_bg = transformed['image1']
        app_img = transformed['image']
        if indicator[1] == 1:
            geo_img, trans_mask = self.perform_geometry_augmentation(app_img, mask, new_bg)
        else:
            geo_img = img
            trans_mask = mask
        return app_img, geo_img, trans_mask

    def random_crop_background(self, image, inpaint, bbox, mask):
        width, height = image.shape[1], image.shape[0]
        bbox_w = float(bbox[2] - bbox[0]) / width
        bbox_h = float(bbox[3] - bbox[1]) / height
        
        inpaint_w, inpaint_h = inpaint.shape[1], inpaint.shape[0]
        left, right, top, down = 0, width, 0, height 
        if bbox_w < self.bbox_maxlen:
            maxcrop = (width - bbox_w * width / self.bbox_maxlen) / 2
            left  = int(np.random.rand() * min(maxcrop, bbox[0]))
            right = width - int(np.random.rand() * min(maxcrop, width - bbox[2]))

        if bbox_h < self.bbox_maxlen:
            maxcrop = (height - bbox_h * height / self.bbox_maxlen) / 2
            top   = int(np.random.rand() * min(maxcrop, bbox[1]))
            down  = height - int(np.random.rand() * min(maxcrop, height - bbox[3]))
        
        trans_bbox = [bbox[0] - left, bbox[1] - top, bbox[2] - left, bbox[3] - top]
        trans_image = image[top:down, left:right]
        trans_inpaint = inpaint[top:down, left:right]
        trans_mask  = mask[top:down, left:right]
        return trans_image, trans_inpaint, trans_bbox, trans_mask
    
    def random_pad_bbox(self, bbox, width, height):
        bbox_pad  = bbox.copy()
        bbox_w = float(bbox[2] - bbox[0]) / width
        bbox_h = float(bbox[3] - bbox[1]) / height
        
        if bbox_w < self.bbox_maxlen:
            maxpad = width * min(self.bbox_maxlen - bbox_w, bbox_w) * 0.5
            bbox_pad[0] = max(0, int(bbox[0] - np.random.rand() * maxpad))
            bbox_pad[2] = min(width, int(bbox[2] + np.random.rand() * maxpad))
        
        if bbox_h < self.bbox_maxlen:
            maxpad = height * min(self.bbox_maxlen - bbox_h, bbox_h) * 0.5
            bbox_pad[1] = max(0, int(bbox[1] - np.random.rand() * maxpad))
            bbox_pad[3] = min(height, int(bbox[3] + np.random.rand() * maxpad))
        return bbox_pad

    
def bbox2mask(bbox, mask_w, mask_h):
    mask = np.zeros((mask_h, mask_w), dtype=np.uint8)
    mask[bbox[1] : bbox[3], bbox[0] : bbox[2]] = 255
    return mask
    
def mask2bbox(mask):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
    if mask.ndim == 3:
        mask = np.squeeze(mask, axis=-1)
    rows = np.any(mask > 0, axis=1)
    cols = np.any(mask > 0, axis=0)
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    return [x1, y1, x2, y2]

    
def constant_pad_bbox(bbox, width, height, value=10):
    ### Get reference image
    bbox_pad=copy.deepcopy(bbox)
    left_space  = bbox[0]
    up_space    = bbox[1]
    right_space = width  - bbox[2]
    down_space  = height - bbox[3] 

    bbox_pad[0]=bbox[0]-min(value, left_space)
    bbox_pad[1]=bbox[1]-min(value, up_space)
    bbox_pad[2]=bbox[2]+min(value, right_space)
    bbox_pad[3]=bbox[3]+min(value, down_space)
    return bbox_pad
    
def rescale_image_with_bbox(image, bbox=None, long_size=1024):
    src_width, src_height = image.size
    if max(src_width, src_height) <= long_size:
        dst_img = image
        dst_width, dst_height = dst_img.size
    else:
        scale = float(long_size) / max(src_width, src_height)
        dst_width, dst_height = int(scale * src_width), int(scale * src_height)
        dst_img  = image.resize((dst_width, dst_height))
    if bbox == None:
        return dst_img
    bbox[0] = int(float(bbox[0]) / src_width  * dst_width)
    bbox[1] = int(float(bbox[1]) / src_height * dst_height)
    bbox[2] = int(float(bbox[2]) / src_width  * dst_width)
    bbox[3] = int(float(bbox[3]) / src_height * dst_height)
    return dst_img, bbox
    
def crop_foreground_by_bbox(img, mask, bbox, pad_bbox=10):
    width,height = img.shape[1], img.shape[0]
    bbox_pad = constant_pad_bbox(bbox, width, height, pad_bbox) if pad_bbox > 0 else bbox
    img = img[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2]]
    if mask is not None:
        mask = mask[bbox_pad[1]:bbox_pad[3],bbox_pad[0]:bbox_pad[2]]
    return img, mask, bbox_pad

def image2inpaint(image, mask):
    if len(mask.shape) == 2:
        mask_f = mask[:,:,np.newaxis]
    else:
        mask_f = mask
    mask_f  = mask_f.astype(np.float32) / 255
    inpaint = image.astype(np.float32)
    gray  = np.ones_like(inpaint) * 127
    inpaint = inpaint * (1 - mask_f) + mask_f * gray
    inpaint = np.uint8(inpaint)
    return inpaint

def check_dir(dir):
    assert os.path.exists(dir), dir
    return dir

def get_bbox_tensor(bbox, width, height):
    norm_bbox = bbox
    norm_bbox = torch.tensor(norm_bbox).reshape(-1).float()
    norm_bbox[0::2] /= width
    norm_bbox[1::2] /= height
    return norm_bbox

    
def reverse_image_tensor(tensor, img_size=(256,256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = (tensor.float() + 1) / 2
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor, (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def reverse_mask_tensor(tensor, img_size=(256,256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def reverse_clip_tensor(tensor, img_size=(256,256)):
    if tensor.ndim == 3:
        tensor = tensor.unsqueeze(0)
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073],  dtype=torch.float)
    MEAN = MEAN.reshape(1, 3, 1, 1).to(tensor.device)
    STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float)
    STD  = STD.reshape(1, 3, 1, 1).to(tensor.device)
    tensor = (tensor * STD) + MEAN
    tensor = torch.clamp(tensor, min=0.0, max=1.0)
    tensor = torch.permute(tensor.float(), (0, 2, 3, 1)) * 255
    tensor = tensor.detach().cpu().numpy()
    img_nps = np.uint8(tensor)
    def np2bgr(img, img_size = img_size):
        if img.shape[:2] != img_size:
            img = cv2.resize(img, img_size)
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_list = [np2bgr(img) for img in img_nps]
    return img_list

def random_crop_image(image, crop_w, crop_h):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    x_space = image.shape[1] - crop_w
    y_space = image.shape[0] - crop_h
    x1 = np.random.randint(0, x_space) if x_space > 0 else 0
    y1 = np.random.randint(0, y_space) if y_space > 0 else 0
    image = image[y1 : y1+crop_h, x1 : x1+crop_w]
    # assert crop.shape[0] == crop_h and crop.shape[1] == crop_w, (y1, x1, image.shape, crop.shape, crop_w, crop_h)
    return image

def read_image(image_path):
    with Image.open(image_path) as img:
        img = img.convert('RGB')
    return img

def read_mask(image_path):
    with Image.open(image_path) as img:
        img = img.convert('L')
    return img

# poisson blending
def poisson_blending(fg, fg_mask, bg, center=None):
    if center is None:
        height, width, _ = bg.shape
        center = (int(width/2), int(height/2))
    return cv2.seamlessClone(fg, bg, fg_mask, center, cv2.MIXED_CLONE)

class OpenImageDataset(data.Dataset):
    def __init__(self,split,**args):
        self.split = ['train', 'validation'] if split == 'train' else [split]
        dataset_dir = args['dataset_dir']
        self.parse_augment_config(args)
        assert os.path.exists(dataset_dir), dataset_dir
        self.bbox_dir = check_dir(os.path.join(dataset_dir, 'refine/box'))
        self.image_dir= check_dir(os.path.join(dataset_dir, 'images'))
        self.inpaint_dir = check_dir(os.path.join(dataset_dir, 'refine/inpaint'))
        self.mask_dir = check_dir(os.path.join(dataset_dir, 'refine/mask'))
        self.bbox_split, self.bbox_list = np.array(self.load_bbox_path_list())
        self.length=len(self.bbox_list)
        self.random_trans = DataAugmentation(self.augment_background, self.augment_box)
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = (args['image_size'], args['image_size'])
        self.sd_transform = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(normalize=False, image_size=(224, 224))
        self.bad_images = []
    
    def load_bbox_path_list(self):
        # cache_dir  = os.path.dirname(os.path.abspath(self.bbox_dir))
        cache_dir = self.bbox_dir
        bbox_list  = []
        bbox_split = [] 
        for split in self.split:
            cache_file = os.path.join(cache_dir, f'{split}.json')
            if os.path.exists(cache_file):
                print('load bbox list from ', cache_file)
                with open(cache_file, 'r') as f:
                    bbox_path_list = json.load(f)
            else:
                bbox_path_list= os.listdir(os.path.join(self.bbox_dir, split))
                bbox_path_list.sort()
                print('save bbox list to ', cache_file)
                with open(cache_file, 'w') as f:
                    json.dump(bbox_path_list, f)
            bbox_split.extend([split] * len(bbox_path_list))
            bbox_list.extend(bbox_path_list)
        return bbox_split, bbox_list

    def parse_augment_config(self, args):
        self.augment_config = args['augment_config'] if 'augment_config' in args else None
        if self.augment_config:
            self.sample_mode = self.augment_config.sample_mode
            self.augment_types = self.augment_config.augment_types
            self.sample_prob = self.augment_config.sample_prob
            if self.sample_mode == 'random':
                assert len(self.augment_types) == len(self.sample_prob), \
                    'len({}) != len({})'.format(self.augment_types, self.sample_prob)
            self.augment_background = self.augment_config.augment_background
            self.augment_box = self.augment_config.augment_box
            self.replace_background_prob = self.augment_config.replace_background_prob
            self.use_inpaint_background  = self.augment_config.use_inpaint_background
        else:
            self.sample_mode   = 'random'
            self.augment_types = [(0,0), (0,1), (1,0), (1,1)]
            self.sample_prob = [1. / len(self.augment_types)] * len(self.augment_types)
            self.augment_background = False
            self.augment_box = False
            self.replace_background_prob = 1
            self.use_inpaint_background  = True
        self.augment_list = list(range(len(self.augment_types)))

    def load_bbox_file(self, bbox_file, split):
        bbox_list = []
        with open(bbox_file, 'r') as f:
            for line in f.readlines():
                info  = line.strip().split(' ')
                bbox  = [int(float(f)) for f in info[:4]]
                mask  = os.path.join(self.mask_dir, split, info[-1])
                inpaint = os.path.join(self.inpaint_dir, split, info[-1].replace('.png', '.jpg'))
                if os.path.exists(mask) and os.path.exists(inpaint):
                    bbox_list.append((bbox, mask, inpaint))
        return bbox_list
    
    def sample_all_augmentations(self, source_np, inpaint_np, bbox, mask, fg_img, fg_mask, new_bg):
        output = {}
        for indicator in self.augment_types:
            sample = self.sample_one_augmentations(source_np, inpaint_np, bbox, mask, fg_img, fg_mask, indicator, new_bg)
            for k,v in sample.items():
                if k not in output:
                    output[k] = [v]
                else:
                    output[k].append(v)
            sample = None
        for k in output.keys():
            output[k] = torch.stack(output[k], dim=0)
        return output

    def sample_one_augmentations(self, source_np, inpaint_np, bbox, mask, fg_img, fg_mask, indicator, new_bg):
        transformed = self.random_trans(source_np, inpaint_np, bbox, mask, 
                                        fg_img, fg_mask, indicator, new_bg)
        # get ground-truth composite image and bbox
        gt_mask = Image.fromarray(transformed["bg_mask"])
        img_width, img_height = gt_mask.size
        gt_mask_tensor = self.mask_transform(gt_mask)
        gt_mask_tensor = torch.where(gt_mask_tensor > 0.5, 1, 0).float() 
        
        gt_img_tensor  = Image.fromarray(transformed['bg_img'])
        gt_img_tensor  = self.sd_transform(gt_img_tensor)
        gt_bbox_tensor = transformed['bbox']
        gt_bbox_tensor = get_bbox_tensor(gt_bbox_tensor, img_width, img_height)
        mask_tensor = Image.fromarray(transformed['pad_mask'])
        mask_tensor = self.mask_transform(mask_tensor)
        mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
        bbox_tensor = transformed['pad_bbox']
        bbox_tensor = get_bbox_tensor(bbox_tensor, img_width, img_height)
        # get foreground and foreground mask
        fg_mask_tensor = Image.fromarray(transformed['fg_mask'])
        fg_mask_tensor = self.clip_mask_transform(fg_mask_tensor)
        fg_mask_tensor = torch.where(fg_mask_tensor > 0.5, 1, 0)
        gt_fg_mask_tensor = Image.fromarray(transformed['gt_fg_mask'])
        gt_fg_mask_tensor = self.clip_mask_transform(gt_fg_mask_tensor)
        gt_fg_mask_tensor = torch.where(gt_fg_mask_tensor > 0.5, 1, 0)
        fg_img_tensor = Image.fromarray(transformed['fg_img'])
        fg_img_tensor = self.clip_transform(fg_img_tensor)
        indicator_tensor = torch.tensor(indicator, dtype=torch.int32)
        # get background image
        if self.use_inpaint_background:
            inpaint = Image.fromarray(transformed['inpaint_img'])
            inpaint = self.sd_transform(inpaint)
        else:
            inpaint = gt_img_tensor * (mask_tensor < 0.5)
        # del transformed
        # transformed = None
        return {"gt_img":  gt_img_tensor,
                "gt_mask": gt_mask_tensor,
                "gt_bbox": gt_bbox_tensor,
                "bg_img": inpaint,
                "bg_mask": mask_tensor,
                "fg_img":  fg_img_tensor,
                "fg_mask": fg_mask_tensor,
                "gt_fg_mask": gt_fg_mask_tensor,
                "bbox": bbox_tensor,
                "indicator": indicator_tensor}

    
    def replace_background_in_foreground(self, fg_img, fg_mask, index):
        bg_idx = int((np.random.randint(1, 100) + index) % self.length)
        bbox_file = self.bbox_list[bg_idx]
        split = self.bbox_split[bg_idx]
        # get source image and mask
        image_path = os.path.join(self.image_dir, split, os.path.splitext(bbox_file)[0] + '.jpg')
        bg_img = read_image(image_path)
        bg_img = rescale_image_with_bbox(bg_img)

        bg_width, bg_height = bg_img.size
        fg_height, fg_width = fg_img.shape[:2]
        if bg_width < fg_width or bg_height < fg_height:
            scale = max(float(fg_width) / bg_width, float(fg_height) / bg_height)            
            bg_width  = int(math.ceil(scale * bg_width))
            bg_height = int(math.ceil(scale * bg_height))
            bg_img = bg_img.resize((bg_width, bg_height), Image.BICUBIC)
        bg_crop = random_crop_image(bg_img, fg_width, fg_height)
        fg_img  = np.where(fg_mask[:,:,np.newaxis] >= 127, fg_img, bg_crop)
        return fg_img, bg_crop
    
    def __getitem__(self, index):
        while True:
            try:
                # get bbox and mask
                bbox_file, split = self.bbox_list[index], self.bbox_split[index] 
                bbox_path = os.path.join(self.bbox_dir, split, bbox_file)
                bbox_info = self.load_bbox_file(bbox_path, split)
                bbox,mask_path,inpaint_path = random.choice(bbox_info)
                # get source image and mask
                image_path = os.path.join(self.image_dir, split, os.path.splitext(bbox_file)[0] + '.jpg')
                source_img  = read_image(image_path)
                source_np   = np.array(source_img)
                mask = read_mask(mask_path)
                mask = mask.resize((source_np.shape[1], source_np.shape[0]))
                mask = np.array(mask)
                # take inpainted image as complete background 
                inpaint_img = read_image(inpaint_path)
                inpaint_img = inpaint_img.resize((source_np.shape[1], source_np.shape[0]), Image.BICUBIC)
                inpaint_np  = np.array(inpaint_img)
                # bbox = mask2bbox(mask)
                fg_img, fg_mask, bbox  = crop_foreground_by_bbox(source_np, mask, bbox)
                if self.replace_background_prob > 0 and np.random.rand() < self.replace_background_prob:
                    fg_img, new_bg = self.replace_background_in_foreground(fg_img, fg_mask, index)
                else:
                    new_bg = None
                # perform data augmentation
                if self.sample_mode == 'random':
                    if max(self.sample_prob) == 1:
                        augment_type = self.sample_prob.index(1)
                    else:
                        augment_type = np.random.choice(self.augment_list, 1, p=self.sample_prob)[0]
                        augment_type = int(augment_type)
                    indicator = self.augment_types[augment_type]
                    sample = self.sample_one_augmentations(source_np, inpaint_np, bbox, mask, fg_img, fg_mask, indicator, new_bg)
                else:
                    sample = self.sample_all_augmentations(source_np, inpaint_np, bbox, mask, fg_img, fg_mask, new_bg)
                sample['image_path'] = image_path
                return sample
            except Exception as e:
                print(os.getpid(), bbox_file, e)
                index = np.random.randint(0, len(self)-1)
        
    def __len__(self):
        return self.length
    
class COCOEEDataset(data.Dataset):
    def __init__(self, **args):
        dataset_dir = args['dataset_dir']
        self.use_inpaint_background = args['augment_config'].use_inpaint_background if 'augment_config' in args else True
        assert os.path.exists(dataset_dir), dataset_dir
        self.src_dir = check_dir(os.path.join(dataset_dir, "GT_3500"))
        self.ref_dir = check_dir(os.path.join(dataset_dir, 'Ref_3500'))
        self.mask_dir = check_dir(os.path.join(dataset_dir, 'Mask_bbox_3500'))
        self.gt_mask_dir = check_dir(os.path.join(dataset_dir, 'mask'))
        self.inpaint_dir = check_dir(os.path.join(dataset_dir, 'inpaint'))
        self.ref_mask_dir = check_dir(os.path.join(dataset_dir, 'ref_mask'))
        self.image_list = os.listdir(self.src_dir)
        self.image_list.sort()
        
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.image_size = args['image_size'], args['image_size']
        self.sd_transform   = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        self.clip_mask_transform = get_tensor(normalize=False, image_size=(224, 224))
        
    def __len__(self):
        return len(self.image_list)
    
    def __getitem__(self, index):
        try:
            image = self.image_list[index]
            src_path = os.path.join(self.src_dir, image)
            src_img = read_image(src_path)
            src_tensor = self.sd_transform(src_img)
            im_name  = os.path.splitext(image)[0].split('_')[0]
            # reference image and object mask
            ref_name = im_name + '_ref.png'
            ref_path = os.path.join(self.ref_dir, ref_name)
            assert os.path.exists(ref_path), ref_path
            ref_img = read_image(ref_path)
            ref_tensor = self.clip_transform(ref_img)
            ref_mask_path = os.path.join(self.ref_mask_dir, ref_name)
            assert os.path.exists(ref_mask_path), ref_mask_path
            ref_mask = read_mask(ref_mask_path)
            ref_mask_tensor = self.clip_mask_transform(ref_mask)
            ref_mask_tensor = torch.where(ref_mask_tensor > 0.5, 1, 0)
            
            mask_path = os.path.join(self.mask_dir, im_name + '_mask.png')
            assert os.path.exists(mask_path), mask_path
            mask_img = read_mask(mask_path)
            mask_img = mask_img.resize((src_img.width, src_img.height))
            bbox = mask2bbox(np.array(mask_img))
            bbox_tensor = get_bbox_tensor(bbox, src_img.width, src_img.height)
            mask_tensor = self.mask_transform(mask_img) 
            mask_tensor = torch.where(mask_tensor > 0.5, 1, 0).float()
            indicator_tensor = torch.tensor([1,1], dtype=torch.int32)
            if self.use_inpaint_background:
                inpaint_path = os.path.join(self.inpaint_dir, image.replace('.png', '.jpg'))
                inpaint_img  = read_image(inpaint_path)
                inpaint_tensor = self.sd_transform(inpaint_img)
            else:
                inpaint_tensor = src_tensor * (1 - mask_tensor)
        
            return {"image_path": src_path,
                    "gt_img":  src_tensor,
                    "bg_img":  inpaint_tensor,
                    "bg_mask": mask_tensor,
                    "fg_img":  ref_tensor,
                    'fg_mask': ref_mask_tensor,
                    "bbox":    bbox_tensor,
                    "indicator": indicator_tensor}
        except:
            idx = np.random.randint(0, len(self)-1)
            return self[idx]

class FOSEDataset(data.Dataset):
    def __init__(self, dataset_dir='/mnt/new/397927/dataset/FOSCom'):
        data_root = dataset_dir
        self.bg_dir   = os.path.join(data_root, 'background')
        self.mask_dir = os.path.join(data_root, 'bbox_mask')
        self.bbox_dir = os.path.join(data_root, 'bbox')
        self.fg_dir   = os.path.join(data_root, 'foreground') 
        self.fgmask_dir = os.path.join(data_root, 'foreground_mask')
        self.image_list = os.listdir(self.bg_dir)
        self.image_size = (512, 512)
        self.clip_transform = get_tensor_clip(image_size=(224, 224))
        self.sd_transform = get_tensor(image_size=self.image_size)
        self.mask_transform = get_tensor(normalize=False, image_size=self.image_size)
        
    def __len__(self):
        return len(self.image_list)

    def load_bbox_file(self, bbox_file):
        bbox_list = []
        with open(bbox_file, 'r') as f:
            for line in f.readlines():
                info  = line.strip().split(' ')
                bbox  = [int(float(f)) for f in info[:4]]
                bbox_list.append(bbox)
        return bbox_list[0]
    
    def __getitem__(self, index):
        image = self.image_list[index]
        bg_path = os.path.join(self.bg_dir, image)
        bg_img  = Image.open(bg_path).convert('RGB')
        bg_w, bg_h = bg_img.size
        bg_t    = self.sd_transform(bg_img)
        fg_path = os.path.join(self.fg_dir, image)
        fg_img  = Image.open(fg_path).convert('RGB')
        fgmask_path = os.path.join(self.fgmask_dir, image)
        fg_mask   = Image.open(fgmask_path).convert('L')

        fg_t     = self.clip_transform(fg_img)
        fgmask_t = self.mask_transform(fg_mask) 
        mask_path = os.path.join(self.mask_dir, image)
        mask = Image.open(mask_path).convert('L')
        mask_t = self.mask_transform(mask)
        mask_t = torch.where(mask_t > 0.5, 1, 0).float()
        inpaint_t = bg_t * (1 - mask_t)
        bbox_path = os.path.join(self.bbox_dir, image.replace('.png', '.txt'))
        bbox   = self.load_bbox_file(bbox_path)
        bbox_t = get_bbox_tensor(bbox, bg_w, bg_h)
        indicator_tensor = torch.tensor([1,1], dtype=torch.int32)

        return {"image_path": bg_path,
                "bg_img":  bg_t,
                "inpaint_img":  inpaint_t,
                "bg_mask": mask_t,
                "fg_img":  fg_t,
                "fg_mask": fgmask_t,
                'bbox': bbox_t,
                "indicator": indicator_tensor}

    
def vis_all_augtypes(batch):
    file = batch['image_path']
    gt_t = batch['gt_img'][0]
    gtmask_t = batch['gt_mask'][0]
    bg_t = batch['bg_img'][0]
    bgmask_t  = batch['bg_mask'][0]
    fg_t = batch['fg_img'][0]
    fgmask_t  = batch['fg_mask'][0]
    indicator = batch['indicator'][0].numpy()
    
    gt_imgs  = reverse_image_tensor(gt_t)
    gt_masks = reverse_mask_tensor(gtmask_t) 
    bg_imgs  = reverse_image_tensor(bg_t)
    bg_masks = reverse_mask_tensor(bgmask_t)
    fg_imgs  = reverse_clip_tensor(fg_t)
    fg_masks = reverse_mask_tensor(fgmask_t)
    
    ver_border = np.ones((gt_imgs[0].shape[0], 10, 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
    img_list = []
    for i in range(len(fg_imgs)):
        text = '[{},{}]'.format(indicator[i][0], indicator[i][1])
        fg_img = fg_imgs[i].copy()
        cv2.putText(fg_img, text, (10,30), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0, 0, 255), 2)
        cat_img = np.concatenate([gt_imgs[i], ver_border, gt_masks[i], ver_border, bg_imgs[i], 
                                  ver_border, fg_img, ver_border, fg_masks[i]], axis=1)
        if i > 0:
            hor_border = np.ones((10, cat_img.shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
            img_list.append(hor_border)
        img_list.append(cat_img)
    img_batch = np.concatenate(img_list, axis=0)
    return img_batch

def vis_random_augtype(batch):
    file = batch['image_path']
    gt_t = batch['gt_img']
    gtmask_t = batch['gt_mask'] if 'gt_mask' in batch else batch['bg_mask']
    bg_t = batch['bg_img']
    bgmask_t  = batch['bg_mask']
    inpaint_t = bg_t * (1-bgmask_t)
    fg_t = batch['fg_img']
    fgmask_t = batch['fg_mask']
    gt_fgmask_t = batch['gt_fg_mask'] if 'gt_fg_mask' in batch else batch['fg_mask']
    indicator = batch['indicator'].numpy()
    
    gt_imgs  = reverse_image_tensor(gt_t)
    gt_masks = reverse_mask_tensor(gtmask_t) 
    bg_imgs  = reverse_image_tensor(bg_t)
    inpaints = reverse_image_tensor(inpaint_t)
    fg_imgs  = reverse_clip_tensor(fg_t)
    fg_masks = reverse_mask_tensor(fgmask_t)
    gt_fgmasks = reverse_mask_tensor(gt_fgmask_t)

    ver_border = np.ones((gt_imgs[0].shape[0], 10, 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
    img_list = []
    for i in range(len(gt_imgs)):
        im_name = os.path.basename(file[i]) if len(file) > 1 else os.path.basename(file[0])
        text = '[{},{}]'.format(indicator[i][0], indicator[i][1])
        fg_img = fg_imgs[i].copy()
        cv2.putText(fg_img, text, (10,30), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # cat_img = np.concatenate([gt_imgs[i], ver_border, gt_masks[i], ver_border, bg_imgs[i], 
        #                           ver_border, fg_img, ver_border, fg_masks[i], gt_fgmasks[i]], axis=1)
        cat_img = np.concatenate([bg_imgs[i], ver_border, fg_img, ver_border, fg_masks[i], ver_border, gt_imgs[i], ver_border, gt_masks[i]], axis=1)
        if i > 0:
            hor_border = np.ones((10, cat_img.shape[1], 3), dtype=np.uint8) * np.array([0, 0, 250]).reshape((1,1,-1))
            img_list.append(hor_border)
        img_list.append(cat_img)
    img_batch = np.concatenate(img_list, axis=0)
    return img_batch
    
def test_cocoee_dataset():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/finetune_paint.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.validation
    dataset  = instantiate_from_config(configs)
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=4, 
                            shuffle=False,
                            num_workers=4)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    vis_dir = os.path.join(proj_dir, 'outputs/test_dataaug/batch_data')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    for i, batch in enumerate(dataloader):
        file = batch['image_path']
        gt_t = batch['gt_img']
        bgmask_t  = batch['bg_mask']
        fg_t = batch['fg_img']
        bbox_t = batch['bbox']
        bg_t = batch['bg_img']
        fg_mask = batch['fg_mask']
        im_name = os.path.basename(file[0])
        print(i, len(dataloader), gt_t.shape, fg_t.shape, gt_t.shape, bbox_t.shape, fg_mask.shape)
        batch_img = vis_random_augtype(batch)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
    

def test_open_images():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/finetune_paint.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.train
    configs.params.split = 'validation'
    configs.params.augment_config.sample_mode = 'all' 
    configs.params.augment_config.augment_types = [(0,0), (1,0), (0,1), (1,1)]
    aug_cfg  = configs.params.augment_config
    dataset  = instantiate_from_config(configs)
    bs = 1 if aug_cfg.sample_mode == 'all' else 4
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=bs, 
                            shuffle=False,
                            num_workers=4)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    vis_dir = os.path.join(proj_dir, 'intermediate_results/test_dataaug/batch_data')
    if os.path.exists(vis_dir):
        shutil.rmtree(vis_dir)
    os.makedirs(vis_dir, exist_ok=True)
    
    for i, batch in enumerate(dataloader):
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor) and batch[k].shape[0] == 1:
                batch[k] = batch[k][0]

        file = batch['image_path']
        gt_t = batch['gt_img']
        gtmask_t = batch['gt_mask']
        bgmask_t  = batch['bg_mask']
        fg_t = batch['fg_img']
        bbox_t = batch['bbox']
        im_name = os.path.basename(file[0])
        # test_fill_mask(batch, i)
        print(i, len(dataloader), gt_t.shape, gtmask_t.shape, fg_t.shape, gt_t.shape, bbox_t.shape)
        batch_img = vis_random_augtype(batch)
        cv2.imwrite(os.path.join(vis_dir, f'batch{i}.jpg'), batch_img)
        if i > 10:
            break
    
def test_open_images_efficiency():
    from omegaconf import OmegaConf
    from ldm.util import instantiate_from_config
    from torch.utils.data import DataLoader
    cfg_path = os.path.join(proj_dir, 'configs/finetune_paint.yaml')
    configs  = OmegaConf.load(cfg_path).data.params.train
    configs.params.split = 'train'
    dataset  = instantiate_from_config(configs)
    bs = 16
    dataloader = DataLoader(dataset=dataset, 
                            batch_size=bs, 
                            shuffle=False,
                            num_workers=128)
    print('{} samples = {} bs x {} batches'.format(
        len(dataset), dataloader.batch_size, len(dataloader)
    ))
    start = time.time()
    data_len = len(dataloader)
    for i,batch in enumerate(tqdm(dataloader)):
        image = batch['gt_img']
        pass

        
if __name__ == '__main__':
    # test_mask_blur_batch()
    test_open_images()
    # test_open_images_efficiency()
    # test_cocoee_dataset()

