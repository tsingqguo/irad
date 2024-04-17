import os
import random
import time
import shutil
import math

import torch
import numpy as np
from torch.optim import SGD, Adam
from tensorboardX import SummaryWriter


class Averager():

    def __init__(self):
        self.n = 0.0
        self.v = 0.0

    def add(self, v, n=1.0):
        self.v = (self.v * self.n + v * n) / (self.n + n)
        self.n += n

    def item(self):
        return self.v


class Timer():

    def __init__(self):
        self.v = time.time()

    def s(self):
        self.v = time.time()

    def t(self):
        return time.time() - self.v


def time_text(t):
    if t >= 3600:
        return '{:.1f}h'.format(t / 3600)
    elif t >= 60:
        return '{:.1f}m'.format(t / 60)
    else:
        return '{:.1f}s'.format(t)


_log_path = None


def set_log_path(path):
    global _log_path
    _log_path = path


def log(obj, filename='log.txt'):
    print(obj)
    if _log_path is not None:
        with open(os.path.join(_log_path, filename), 'a') as f:
            print(obj, file=f)


def ensure_path(path, remove=True):
    basename = os.path.basename(path.rstrip('/'))
    if os.path.exists(path):
        if remove and (basename.startswith('_')
                or input('{} exists, remove? (y/[n]): '.format(path)) == 'y'):
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


def set_save_path(save_path, remove=True):
    ensure_path(save_path, remove=remove)
    set_log_path(save_path)
    writer = SummaryWriter(os.path.join(save_path, 'tensorboard'))
    return log, writer


def compute_num_params(model, text=False):
    tot = int(sum([np.prod(p.shape) for p in model.parameters()]))
    if text:
        if tot >= 1e6:
            return '{:.1f}M'.format(tot / 1e6)
        else:
            return '{:.1f}K'.format(tot / 1e3)
    else:
        return tot


def make_optimizer(param_list, optimizer_spec, load_sd=False):
    Optimizer = {
        'sgd': SGD,
        'adam': Adam
    }[optimizer_spec['name']]
    optimizer = Optimizer(param_list, **optimizer_spec['args'])
    if load_sd:
        optimizer.load_state_dict(optimizer_spec['sd'])
    return optimizer


def make_coord(shape, ranges=None, flatten=True):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_coord_shift(shape, ranges=None, flatten=True, shift_scale=1):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    r = 0
    torch.set_printoptions(precision=8)
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    r = r * shift_scale
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            ret[i][j][0] = ret[i][j][0] + r
            ret[i][j][1] = ret[i][j][1] + r
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_coord_shift_reg(shape, ranges=None, flatten=True, shift_scale=1):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    r = 0
    torch.set_printoptions(precision=8)
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    r = r * shift_scale
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            ret[i][j][0] = ret[i][j][0] + r
            ret[i][j][1] = ret[i][j][1] + r
            if ret[i][j][0] > 1:
                ret[i][j][0] = 1
            if ret[i][j][0] < -1:
                ret[i][j][0] = -1
            if ret[i][j][1] > 1:
                ret[i][j][1] = 1
            if ret[i][j][1] < -1:
                ret[i][j][1] = -1
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_coord_normal(shape, ranges=None, flatten=True, shift_scale=1):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    r = 0
    torch.set_printoptions(precision=8)
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    r = r * shift_scale
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            normal_r = random.normalvariate(0, r)
            ret[i][j][0] = ret[i][j][0] + normal_r
            ret[i][j][1] = ret[i][j][1] + normal_r
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_coord_normal2(shape, ranges=None, flatten=True, shift_scale=1):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    r = 0
    torch.set_printoptions(precision=8)
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    r = r * shift_scale
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            normal_r1 = random.normalvariate(0, r)
            normal_r2 = random.normalvariate(0, r)
            ret[i][j][0] = ret[i][j][0] + normal_r1
            ret[i][j][1] = ret[i][j][1] + normal_r2
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def make_coord_uniform_random(shape, ranges=None, flatten=True, shift_scale=1):
    """ Make coordinates at grid centers.
    """
    coord_seqs = []
    r = 0
    torch.set_printoptions(precision=8)
    for i, n in enumerate(shape):
        if ranges is None:
            v0, v1 = -1, 1
        else:
            v0, v1 = ranges[i]
        r = (v1 - v0) / (2 * n)
        seq = v0 + r + (2 * r) * torch.arange(n).float()
        coord_seqs.append(seq)
    ret = torch.stack(torch.meshgrid(*coord_seqs), dim=-1)
    r = r * shift_scale
    for i in range(len(ret)):
        for j in range(len(ret[i])):
            r1 = random.uniform(0, r)
            r2 = random.uniform(0, r)
            ret[i][j][0] = ret[i][j][0] + r1
            ret[i][j][1] = ret[i][j][1] + r2
    if flatten:
        ret = ret.view(-1, ret.shape[-1])
    return ret


def to_pixel_samples(img):
    """ Convert the image to coord-RGB pairs.
        img: Tensor, (3, H, W)
    """
    coord = make_coord(img.shape[-2:])
    rgb = img.view(3, -1).permute(1, 0)
    return coord, rgb


def calc_psnr(sr, hr, dataset=None, scale=1, rgb_range=1):
    diff = (sr - hr) / rgb_range
    if dataset is not None:
        if dataset == 'benchmark':
            shave = scale
            if diff.size(1) > 1:
                gray_coeffs = [65.738, 129.057, 25.064]
                convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
                diff = diff.mul(convert).sum(dim=1)
        elif dataset == 'div2k':
            shave = scale + 6
        else:
            raise NotImplementedError
        valid = diff[..., shave:-shave, shave:-shave]
    else:
        valid = diff
    mse = valid.pow(2).mean()
    return -10 * torch.log10(mse)
