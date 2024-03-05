import argparse, os, sys, glob
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
import time
from pytorch_lightning import seed_everything
try:
    from lightning_fabric.utilities.seed import log
    log.propagate = False
except:
    pass
from torch import device
import torchvision

proj_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(proj_dir)
sys.path.insert(0, proj_dir)

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from torchvision.transforms import Resize
from ldm.data.open_images_control import get_tensor, get_tensor_clip, get_bbox_tensor, bbox2mask, mask2bbox

def clip2sd(x):
    # clip input tensor to  stable diffusion tensor
    MEAN = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(1,-1,1,1).to(x.device)
    STD  = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(1,-1,1,1).to(x.device)
    denorm = x * STD + MEAN
    sd_x = denorm * 2 - 1
    return sd_x

def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]
    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print('load checkpoint {}'.format(ckpt))
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    model.eval()
    return model

clip_transform = get_tensor_clip(image_size=(224, 224))
sd_transform   = get_tensor(image_size=(512, 512))
mask_transform = get_tensor(normalize=False, image_size=(512, 512))

def generate_image_batch(bg_path, fg_path, bbox, fg_mask_path=None):
    bg_img     = Image.open(bg_path).convert('RGB')
    bg_w, bg_h = bg_img.size
    bg_t       = sd_transform(bg_img)
    fg_img     = Image.open(fg_path).convert('RGB')
    if fg_mask_path != None:
        fg_mask= Image.open(fg_mask_path).convert('RGB')
        fg_mask= fg_mask.resize((fg_img.width, fg_img.height))
        black  = np.zeros_like(fg_mask)
        fg_mask= np.asarray(fg_mask)
        fg_img = np.asarray(fg_img)
        fg_img = np.where(fg_mask > 127, fg_img, black)
        fg_img = Image.fromarray(fg_img)
    fg_t       = clip_transform(fg_img)
    mask       = Image.fromarray(bbox2mask(bbox, bg_w, bg_h))
    mask_t     = mask_transform(mask)
    mask_t     = torch.where(mask_t > 0.5, 1, 0).float()
    inpaint_t  = bg_t * (1 - mask_t)
    bbox_t     = get_bbox_tensor(bbox, bg_w, bg_h)
    return {"bg_img":  inpaint_t.unsqueeze(0),
            "bg_mask": mask_t.unsqueeze(0),
            "fg_img":  fg_t.unsqueeze(0),
            "bbox":    bbox_t.unsqueeze(0)
            }

def prepare_input(batch, model, shape, device, num_samples, indicator):
    if num_samples > 1:
        for k in batch.keys():
            if isinstance(batch[k], torch.Tensor):
                batch[k] = torch.cat([batch[k]] * num_samples, dim=0)
    test_model_kwargs={}
    bg_img    = batch['bg_img'].to(device)
    bg_latent = model.encode_first_stage(bg_img)
    bg_latent = model.get_first_stage_encoding(bg_latent).detach()
    test_model_kwargs['bg_latent'] = bg_latent
    rs_mask = F.interpolate(batch['bg_mask'].to(device), shape[-2:])
    rs_mask = torch.where(rs_mask > 0.5, 1.0, 0.0)
    test_model_kwargs['bg_mask']  = rs_mask
    test_model_kwargs['bbox']  = batch['bbox'].to(device)
    fg_tensor = batch['fg_img'].to(device)
    c = model.get_learned_conditioning((fg_tensor, None))
    indicator = torch.tensor(indicator).int().to(device)
    if num_samples // len(indicator) > 1:
        indicator = indicator.repeat_interleave(num_samples // len(indicator), dim=0)
    c.append(indicator)
    c.append(torch.tensor([True] * indicator.shape[0])) 
    uc_global = model.learnable_vector.repeat(c[0].shape[0], 1, 1) # 1,1,768
    if hasattr(model, 'get_unconditional_local_embedding'):
        uc_local = model.get_unconditional_local_embedding(c[1])
    else:
        uc_local = c[1]
    uc = [uc_global, uc_local, indicator, torch.tensor([False] * indicator.shape[0])]
    return test_model_kwargs, c, uc

def tensor2numpy(image, normalized=False, image_size=(512, 512)):
    image = Resize(image_size, antialias=True)(image)
    if not normalized:
        image = (image + 1.0) / 2.0  # -1,1 -> 0,1; b,c,h,w
    image = torch.clamp(image, 0., 1.)
    if image.dim() == 3:
        image = image.unsqueeze(0)
    image = image.permute(0, 2, 3, 1)
    image = image.numpy()
    image = (image * 255).astype(np.uint8)
    return image

def save_image(img, img_path):
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flag = cv2.imwrite(img_path, img)
    if not flag:
        print(img_path, img.shape)

def draw_bbox_on_background(image_nps, norm_bbox, color=(255,215,0), thickness=3):
    dst_list = []
    for i in range(image_nps.shape[0]):
        img = image_nps[i].copy()
        h,w,_ = img.shape
        x1 = int(norm_bbox[0,0] * w)
        y1 = int(norm_bbox[0,1] * h)
        x2 = int(norm_bbox[0,2] * w)
        y2 = int(norm_bbox[0,3] * h)
        dst = cv2.rectangle(img, (x1, y1), (x2, y2), color=color, thickness=thickness)
        dst_list.append(dst)
    dst_nps = np.stack(dst_list, axis=0)
    return dst_nps

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--testdir",
        type=str,
        help="background image path",
        default="./examples"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="results"
    )
    parser.add_argument(
        "--skip_grid",
        action='store_true',
        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples",
    )
    parser.add_argument(
        "--sample_steps",
        type=int,
        default=50,
        help="number of sampling steps",
    )
    parser.add_argument(
        "--gpu",
        type=int,
        help="gpu id",
        default=0
    )
    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )
    parser.add_argument(
        "--fixed_code",
        type=bool,
        default=False,
        help="if enabled, uses the same starting code across samples ",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="blending",
        help="type of controllabel composition, choose from [blending, harmonization, viewsynthesis, composition]",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="how many samples to produce for each given reference image. A.k.a. batch size",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="./configs/controlcom.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt_dir",
        default="./checkpoints",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=321,
        help="the seed (for reproducible sampling)",
    )
    opt = parser.parse_args()
    return opt

def parse_input_bbox(opt, img_name):
    mask_img = os.path.join(opt.testdir, 'mask_bbox', img_name)
    mask_txt = os.path.join(opt.testdir, 'bbox', img_name.split('.')[0] + '.txt')
    if os.path.exists(mask_img):
        bbox = mask2bbox(Image.open(mask_img))
    elif os.path.exists(mask_txt):
        bbox = txt2bbox(mask_txt)
    else:
        raise Exception('undefined bbox format')
    return bbox

def txt2bbox(mask_txt):
    with open(mask_txt, 'r') as fb:
        box = fb.readline().strip().split(' ')[:4]
        box = list(map(int, box))
        return box
    
def generate_image_grid(batch, comp_img):
    res_dict = {} 
    img_size = (512, 512)
    res_dict['bg']   = tensor2numpy(batch['bg_img'], image_size=img_size)
    res_dict['bbox'] = draw_bbox_on_background(res_dict['bg'], batch['bbox'], color=(255,215,0), thickness=3)
    res_dict['fg']   = tensor2numpy(clip2sd(batch['fg_img']), image_size=img_size)
    res_dict['comp'] = comp_img
    x_border = (np.ones((img_size[0], 10, 3)) * 255).astype(np.uint8)
    grid_img  = []
    grid_row = [res_dict['bbox'][0], x_border, res_dict['fg'][0]]
    for i in range(comp_img.shape[0]):
        comp_img =  res_dict['comp'][i]
        grid_row += [x_border, comp_img]
    grid_img = np.concatenate(grid_row, axis=1)
    grid_img = Image.fromarray(grid_img)
    return grid_img
    


if __name__ == "__main__":
    opt = argument_parse()
    if opt.task in ['blending', 'harmonization']:
        weight_path = os.path.join(opt.ckpt_dir, 'ControlCom_blend_harm.pth')
        indicator   = [[0,0]] if opt.task == 'blending' else [[1,0]]
    else:
        weight_path = os.path.join(opt.ckpt_dir, 'ControlCom_view_comp.pth')
        indicator   = [[0,1]] if opt.task == 'viewsynthesis' else [[1,1]]
    assert os.path.exists(weight_path), weight_path 
    config      = OmegaConf.load(opt.config)
    clip_path   = os.path.join(opt.ckpt_dir, 'openai-clip-vit-large-patch14')
    assert os.path.exists(clip_path), clip_path
    config.model.params.cond_stage_config.params.version = clip_path
    model       = load_model_from_config(config, weight_path)
    device      = torch.device(f'cuda:{opt.gpu}')
    model       = model.to(device)
    if opt.plms:
        print(f'Using PLMS samplers with {opt.sample_steps} sampling steps')
        sampler = PLMSSampler(model)
    else:
        print(f'Using DDIM samplers with {opt.sample_steps} sampling steps')
        sampler = DDIMSampler(model)
    
    img_size    = (512, 512)
    shape = [4, img_size[1] // 8, img_size[0] // 8]
    sample_steps = opt.sample_steps
    num_samples  = opt.num_samples
    guidance_scale = opt.scale
    if opt.fixed_code:
        seed_everything(opt.seed)
    start_code = torch.randn([num_samples]+shape, device=device)
    # start_code = start_code.repeat(len(indicator), 1, 1, 1)
    test_list  = os.listdir(os.path.join(opt.testdir, 'background'))
    print('find {} pairs of test samples'.format(len(test_list)))
    os.makedirs(opt.outdir, exist_ok=True)
    
    for img_name in test_list:
        bg_path    = os.path.join(opt.testdir, 'background', img_name)
        fg_path    = os.path.join(opt.testdir, 'foreground', img_name)
        bbox       = parse_input_bbox(opt, img_name)
        fgmask_path= os.path.join(opt.testdir, 'foreground_mask', img_name)
        fgmask_path= None if not os.path.exists(fgmask_path) else fgmask_path
        start = time.time()
        batch = generate_image_batch(bg_path, fg_path, bbox, fgmask_path)
        test_model_kwargs, c, uc = prepare_input(batch, model, shape, device, num_samples*len(indicator), indicator)
        samples_ddim, _ = sampler.sample(S=sample_steps,
                                        conditioning=c,
                                        batch_size=num_samples*len(indicator),
                                        shape=shape,
                                        verbose=False,
                                        eta=0.0,
                                        x_T=start_code,
                                        unconditional_guidance_scale=guidance_scale,
                                        unconditional_conditioning=uc,
                                        test_model_kwargs=test_model_kwargs)
        x_samples_ddim = model.decode_first_stage(samples_ddim[:,:4]).cpu().float()
        print('inference time: {:.1f}s'.format(time.time() - start))
        comp_img = tensor2numpy(x_samples_ddim, image_size=img_size)
        # save composite results
        for i in range(comp_img.shape[0]):
            if i > 0:
                res_path = os.path.join(opt.outdir, img_name.split('.')[0] + f'_{opt.task}_sample{i}.png')
            else:
                res_path = os.path.join(opt.outdir, img_name.split('.')[0] + f'_{opt.task}.jpg')
            save_image(comp_img[i], res_path)
            print('save result to {}'.format(res_path))
        if not opt.skip_grid:
            grid_img  = generate_image_grid(batch, comp_img)
            grid_path = os.path.join(opt.outdir, img_name.split('.')[0] + f'_{opt.task}_grid.jpg')
            save_image(grid_img, grid_path)
            print('save grid_result to {}'.format(grid_path))